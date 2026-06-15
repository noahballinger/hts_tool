import streamlit as st
import pandas as pd
import os
import sqlite3
import time
from datetime import datetime
from functools import wraps

# ---------------------------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Team HTS Classifier", layout="wide")

MATCH_FILE   = 'Odoo_HS_Codes_Final_Matched.csv'
PRODUCT_FILE = 'Odoo_Products.csv'
VENDOR_FILE  = 'Vendor HTS Codes.csv'
DB_FILE      = 'classification_progress.db'

# ---------------------------------------------------------------------------
# Multi-user: require a display name before the app renders anything else.
# This is stored only in the browser session – no passwords, no server state.
# ---------------------------------------------------------------------------
def require_username():
    if not st.session_state.get("username"):
        st.sidebar.title("Welcome")
        name = st.sidebar.text_input("Enter your name to begin:")
        if st.sidebar.button("Start Session") and name.strip():
            st.session_state.username = name.strip()
            st.rerun()
        st.sidebar.info("Please enter your name to begin classifying.")
        st.stop()

require_username()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db_connection():
    """
    WAL mode allows concurrent readers + one writer without blocking.
    timeout=30 makes writers queue rather than crash on contention.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_retry(max_attempts: int = 4, base_delay: float = 0.3):
    """
    Exponential-backoff retry for transient SQLite lock errors.
    Covers the window where two users write at exactly the same millisecond.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() and attempt < max_attempts - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator


def init_db():
    """
    Schema includes `classified_by` / `approved_by` so every action is
    attributable to a team member.  Uses ALTER TABLE to add columns if an
    older DB already exists without them.
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS category_approvals (
                odoo_hs     TEXT PRIMARY KEY,
                approved_hts TEXT,
                approved_by  TEXT,
                approved_at  TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS product_classifications (
                product_id   TEXT PRIMARY KEY,
                product_name TEXT,
                old_hs       TEXT,
                new_hts      TEXT,
                classified_by TEXT,
                classified_at TEXT
            )
        ''')
        # Non-destructive migration for existing DBs that lack the new columns
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(category_approvals)")}
        for col, typedef in [("approved_by", "TEXT"), ("approved_at", "TEXT")]:
            if col not in existing_cols:
                c.execute(f"ALTER TABLE category_approvals ADD COLUMN {col} {typedef}")

        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(product_classifications)")}
        for col, typedef in [("classified_by", "TEXT")]:
            if col not in existing_cols:
                c.execute(f"ALTER TABLE product_classifications ADD COLUMN {col} {typedef}")

        conn.commit()

init_db()

# ---------------------------------------------------------------------------
# Static data (CSV files never change at runtime → no TTL needed)
# ---------------------------------------------------------------------------
@st.cache_data
def load_source_data():
    matches_df = pd.read_csv(MATCH_FILE)
    matches_df['HS Code'] = (
        matches_df['HS Code']
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )
    matches_df = matches_df.drop_duplicates(subset=['HS Code'])
    matches_df['HTS_Options'] = matches_df['All Vendor Matches'].apply(
        lambda x: list({
            str(c).strip()
            for c in str(x).split(',')
            if str(c).strip() and str(c).strip() != 'nan'
        })
    )

    if os.path.exists(PRODUCT_FILE):
        products_df = pd.read_csv(PRODUCT_FILE)
    else:
        products_df = pd.DataFrame({
            'Product_ID':      ['SKU-001', 'SKU-002', 'SKU-003'],
            'Product_Name':    ['Castor Oil 500ml', 'Botanical Shampoo', 'Generic Prep'],
            'Current_Odoo_HS': ['151530', '330510', '210690'],
        })

    desc_map = {}
    if os.path.exists(VENDOR_FILE):
        vendor_df = pd.read_csv(VENDOR_FILE)
        vendor_df['Tariff'] = vendor_df['Tariff'].astype(str).str.replace(r'\D', '', regex=True)
        for _, row in vendor_df.iterrows():
            tariff    = str(row['Tariff']).strip()
            g_desc    = str(row.get('Goods Description', '')).strip() if pd.notna(row.get('Goods Description')) else ''
            o_desc    = str(row.get('Official Description', '')).strip() if pd.notna(row.get('Official Description')) else ''
            desc_map[tariff] = f"{g_desc} | {o_desc}" if g_desc and o_desc else (g_desc or o_desc)

    return matches_df, products_df, desc_map


matches_df, products_df, desc_map = load_source_data()


def make_label(hts_code: str) -> str:
    return f"{hts_code} | {desc_map.get(str(hts_code), 'No vendor description found')}"


# ---------------------------------------------------------------------------
# Live DB reads – short TTL (15 s) so every user sees teammates' changes
# quickly without hammering the DB on every interaction.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15)
def get_category_approvals() -> dict:
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM category_approvals", conn)
    return df.set_index('odoo_hs').to_dict('index') if not df.empty else {}


@st.cache_data(ttl=15)
def get_classified_products() -> pd.DataFrame:
    with get_db_connection() as conn:
        return pd.read_sql_query(
            "SELECT product_id, classified_by, classified_at FROM product_classifications",
            conn
        )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Classification Workflow")
st.sidebar.markdown(f"👤 **Session:** {st.session_state.username}")
step = st.sidebar.radio("Go to:", [
    "1. Category Review",
    "2. Product Review",
    "3. Export & Progress",
])
if st.sidebar.button("🔄 Refresh team data"):
    get_category_approvals.clear()
    get_classified_products.clear()
    st.rerun()

# ===========================================================================
# STEP 1 – CATEGORY REVIEW
# ===========================================================================
if step == "1. Category Review":
    st.title("Step 1: Confirm Category Matches")
    st.markdown("Select the accurate **10-digit HTS Codes** for each Odoo category family.")

    approvals    = get_category_approvals()
    existing_cats = {k: v.get('approved_hts', '') for k, v in approvals.items()}
    approval_meta = {k: (v.get('approved_by', ''), v.get('approved_at', '')) for k, v in approvals.items()}

    with st.form("category_form"):
        for _, row in matches_df.iterrows():
            odoo_hs = str(row['HS Code'])
            options  = row['HTS_Options']
            if not options:
                continue

            col_title, col_meta = st.columns([3, 1])
            with col_title:
                st.subheader(f"`{odoo_hs}` – {row['Description']}")
            with col_meta:
                if approval_meta.get(odoo_hs, ('', ''))[0]:
                    by, at = approval_meta[odoo_hs]
                    st.caption(f"✅ Saved by **{by}** on {at[:10]}")

            saved_val        = existing_cats.get(odoo_hs, '')
            saved_list       = [x.strip() for x in saved_val.split(',') if x.strip()]
            default_selection = [x for x in saved_list if x in options] or options

            st.multiselect(
                "Valid HTS Classifications:",
                options=options,
                default=default_selection,
                format_func=make_label,
                key=f"cat_{odoo_hs}",
            )
            st.divider()

        if st.form_submit_button("Save All Category Approvals", type="primary"):
            @db_retry()
            def _save_categories():
                with get_db_connection() as conn:
                    c   = conn.cursor()
                    now = datetime.now().isoformat()
                    for _, mrow in matches_df.iterrows():
                        hs  = str(mrow['HS Code'])
                        sel = st.session_state.get(f"cat_{hs}", [])
                        c.execute(
                            '''INSERT OR REPLACE INTO category_approvals
                               (odoo_hs, approved_hts, approved_by, approved_at)
                               VALUES (?, ?, ?, ?)''',
                            (hs, ", ".join(sel), st.session_state.username, now),
                        )
                    conn.commit()

            _save_categories()
            get_category_approvals.clear()
            st.success(f"✅ Categories saved by **{st.session_state.username}**! Proceed to Step 2.")

# ===========================================================================
# STEP 2 – PRODUCT REVIEW (collaborative, race-condition safe)
# ===========================================================================
elif step == "2. Product Review":
    st.title("Step 2: Product Classification")

    approvals       = get_category_approvals()
    cat_mappings    = {k: v.get('approved_hts', '') for k, v in approvals.items()}
    classified_df   = get_classified_products()
    classified_ids  = classified_df['product_id'].tolist() if not classified_df.empty else []

    if not cat_mappings:
        st.warning("Please complete Step 1 (Category Review) to populate mapping choices first.")
        st.stop()

    remaining = products_df[~products_df['Product_ID'].astype(str).isin(classified_ids)]
    total     = len(products_df)
    completed = total - len(remaining)

    st.progress(completed / total if total > 0 else 1.0)
    st.write(f"Progress: **{completed}** / **{total}** items classified.")

    # Recent team activity feed
    if not classified_df.empty:
        with st.expander("📋 Recent team activity"):
            with get_db_connection() as conn:
                feed = pd.read_sql_query(
                    '''SELECT product_name, new_hts, classified_by, classified_at
                       FROM product_classifications
                       ORDER BY classified_at DESC LIMIT 8''',
                    conn,
                )
            st.dataframe(feed, use_container_width=True)

    if remaining.empty:
        st.success("🎉 All products have been fully classified by the team!")
        st.stop()

    row        = remaining.iloc[0]
    product_id = str(row['Product_ID'])
    current_hs = str(row['Current_Odoo_HS'])

    st.header(row['Product_Name'])
    st.markdown(f"**SKU:** `{product_id}` | **Category context:** `{current_hs}`")

    options = [x.strip() for x in cat_mappings.get(current_hs, '').split(',') if x.strip()]
    options.append("Other (Manual Entry)")

    with st.form("product_form", clear_on_submit=True):
        choice       = st.radio(
            "Select 10-Digit HTS Code:",
            options=options,
            format_func=lambda x: x if x == "Other (Manual Entry)" else make_label(x),
        )
        manual_entry = st.text_input("If 'Other', enter custom 10-digit code:")

        if st.form_submit_button("Confirm & Next Item", type="primary"):
            final_code = manual_entry.strip() if choice == "Other (Manual Entry)" else choice

            @db_retry()
            def _save_product():
                with get_db_connection() as conn:
                    c = conn.cursor()
                    # Guard: check inside the transaction so two users
                    # can't both claim the same product simultaneously.
                    already = c.execute(
                        "SELECT product_id FROM product_classifications WHERE product_id = ?",
                        (product_id,),
                    ).fetchone()
                    if already:
                        return False
                    c.execute(
                        '''INSERT INTO product_classifications
                           (product_id, product_name, old_hs, new_hts, classified_by, classified_at)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (product_id, row['Product_Name'], current_hs, final_code,
                         st.session_state.username, datetime.now().isoformat()),
                    )
                    conn.commit()
                    return True

            saved = _save_product()
            get_classified_products.clear()

            if not saved:
                st.warning("⚡ A teammate just classified this item — skipping to the next one.")
            st.rerun()

# ===========================================================================
# STEP 3 – EXPORT & AUDIT
# ===========================================================================
elif step == "3. Export & Progress":
    st.title("Step 3: Export & Audit Results")

    with get_db_connection() as conn:
        results_df = pd.read_sql_query(
            "SELECT * FROM product_classifications ORDER BY classified_at DESC", conn
        )

    if results_df.empty:
        st.info("No classifications committed yet.")
        st.stop()

    st.subheader("Team Contributions")
    contribs = results_df.groupby('classified_by').size().reset_index(name='items_classified')
    st.dataframe(contribs, use_container_width=True)

    st.subheader("Full Classification Log")
    st.dataframe(results_df, use_container_width=True)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Master Classification CSV",
        data=csv,
        file_name="Final_Classified_Products.csv",
        mime='text/csv',
    )