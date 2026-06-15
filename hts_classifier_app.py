import streamlit as st
import pandas as pd
import psycopg2
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Team HTS Classifier", layout="wide")


# ---------------------------------------------------------------------------
# Simple Password Authentication
# ---------------------------------------------------------------------------
def require_password():
    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Team HTS Classifier")

    password = st.text_input(
        "Password",
        type="password",
        key="password_input"
    )

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password")

    st.stop()

require_password()

# ---------------------------------------------------------------------------
# Multi-user: require a display name before anything renders.
# ---------------------------------------------------------------------------
def require_username():
    if not st.session_state.get("username"):
        st.sidebar.title("Welcome")
        name = st.sidebar.text_input("Enter your name to begin:")
        if st.sidebar.button("Start Session") and name.strip():
            st.session_state.username = name.strip()
            st.rerun()
        st.sidebar.info("Please enter your name to start classifying.")
        st.stop()

require_username()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(st.secrets["DATABASE_URL"])
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_retry(max_attempts: int = 4, base_delay: float = 0.3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError):
                    if attempt < max_attempts - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator


def init_db():
    """Creates app tables if they don't exist. Safe to run on every startup."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS category_approvals (
                odoo_hs      TEXT PRIMARY KEY,
                approved_hts TEXT,
                approved_by  TEXT,
                approved_at  TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS product_classifications (
                product_id    TEXT PRIMARY KEY,
                product_name  TEXT,
                old_hs        TEXT,
                new_hts       TEXT,
                classified_by TEXT,
                classified_at TEXT
            )
        ''')

init_db()

# ---------------------------------------------------------------------------
# Source data — now read from Supabase tables instead of CSV files.
# Cached forever: this data never changes during a session.
# ---------------------------------------------------------------------------
@st.cache_data
def load_source_data():
    with get_db_connection() as conn:
        matches_df  = pd.read_sql_query("SELECT * FROM hs_matches",   conn)
        products_df = pd.read_sql_query("SELECT * FROM products",      conn)
        vendor_df   = pd.read_sql_query("SELECT * FROM vendor_codes",  conn)
    
    print("RAW COLUMNS")
    print(matches_df.columns.tolist())

    # --- Normalise the matches table ---
    # Find the HS code column (Supabase may lowercase headers)
    hs_col   = next(c for c in matches_df.columns if c.lower().replace(' ','') in ('hscode','hs_code'))
    desc_col = next(c for c in matches_df.columns if 'desc' in c.lower())
    all_col = "All Vendor Matches"


    matches_df = matches_df.rename(columns={hs_col: 'HS Code', desc_col: 'Description', all_col: 'All Vendor Matches'})
    matches_df['HS Code'] = (
        matches_df['HS Code']
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.strip()
    )
    matches_df = matches_df.drop_duplicates(subset=['HS Code'])
    matches_df = matches_df.reset_index(drop=True)
    print(matches_df.columns.tolist())
    print(matches_df.columns[matches_df.columns.duplicated()].tolist())
    matches_df['HTS_Options'] = matches_df['All Vendor Matches'].apply(
        lambda x: list({
            str(c).strip()
            for c in str(x).split(',')
            if str(c).strip() and str(c).strip() != 'nan'
        })
    )

    # --- Normalise the vendor table into a description lookup dict ---
    tariff_col = next(c for c in vendor_df.columns if 'tariff' in c.lower())
    goods_col  = next((c for c in vendor_df.columns if 'goods' in c.lower()), None)
    off_col    = next((c for c in vendor_df.columns if 'official' in c.lower()), None)

    desc_map = {}
    for _, row in vendor_df.iterrows():
        tariff = str(row[tariff_col]).strip().replace('.0', '')
        g = str(row[goods_col]).strip() if goods_col and pd.notna(row[goods_col]) else ''
        o = str(row[off_col]).strip()   if off_col  and pd.notna(row[off_col])   else ''
        desc_map[tariff] = f"{g} | {o}" if g and o else (g or o)

    return matches_df, products_df, desc_map


matches_df, products_df, desc_map = load_source_data()



def make_label(hts_code: str) -> str:
    return f"{hts_code} | {desc_map.get(str(hts_code), 'No vendor description found')}"

# Products table schema from Supabase
products_df = products_df.rename(columns={
    'Internal Reference': 'Product_ID',
    'Name': 'Product_Name',
    'HS Code': 'Current_Odoo_HS'
})

# ---------------------------------------------------------------------------
# Live DB reads — 15-second TTL so teammates' changes appear quickly
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
            "SELECT product_id, classified_by, classified_at FROM product_classifications", conn
        )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Classification Workflow")
st.sidebar.markdown(f"👤 **Session:** {st.session_state.username}")
step = st.sidebar.radio("Go to:", [
    "1. Category Review",
    "2. Product Review",
    "2B. Batch Classification",
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

    approvals     = get_category_approvals()
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
                st.subheader(f"`{odoo_hs}` - {row['Description']}")
            with col_meta:
                if approval_meta.get(odoo_hs, ('', ''))[0]:
                    by, at = approval_meta[odoo_hs]
                    st.caption(f"✅ Saved by **{by}** on {at[:10]}")

            saved_val         = existing_cats.get(odoo_hs, '')
            saved_list        = [x.strip() for x in saved_val.split(',') if x.strip()]
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
                            '''
                            INSERT INTO category_approvals (odoo_hs, approved_hts, approved_by, approved_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (odoo_hs) DO UPDATE SET
                                approved_hts = EXCLUDED.approved_hts,
                                approved_by  = EXCLUDED.approved_by,
                                approved_at  = EXCLUDED.approved_at
                            ''',
                            (hs, ", ".join(sel), st.session_state.username, now),
                        )

            _save_categories()
            get_category_approvals.clear()
            st.success(f"✅ Categories saved by **{st.session_state.username}**! Proceed to Step 2.")

# ===========================================================================
# STEP 2 – PRODUCT REVIEW
# ===========================================================================
elif step == "2. Product Review":
    st.title("Step 2: Product Classification")

    approvals      = get_category_approvals()
    cat_mappings   = {k: v.get('approved_hts', '') for k, v in approvals.items()}
    classified_df  = get_classified_products()
    classified_ids = classified_df['product_id'].tolist() if not classified_df.empty else []

    if not cat_mappings:
        st.warning("Please complete Step 1 (Category Review) to populate mapping choices first.")
        st.stop()

    remaining = products_df[~products_df['Product_ID'].astype(str).isin(classified_ids)]
    total     = len(products_df)
    completed = total - len(remaining)

    st.progress(completed / total if total > 0 else 1.0)
    st.write(f"Progress: **{completed}** / **{total}** items classified.")

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

        choice = st.radio(
            "Select 10-Digit HTS Code:",
            options=options,
            format_func=lambda x: x if x == "Other (Manual Entry)" else make_label(x),
        )

        st.markdown("### Or Search All Available HTS Codes")

        all_hts_codes = sorted(desc_map.keys())

        searched_code = st.selectbox(
            "Search HTS Codes",
            options=[""] + all_hts_codes,
            format_func=lambda x: (
                "Search for an HTS code..."
                if x == ""
                else make_label(x)
            ),
        )

        manual_entry = st.text_input(
            "Custom HTS Code (optional)"
        )

        if st.form_submit_button("Confirm & Next Item", type="primary"):

            if manual_entry.strip():
                final_code = manual_entry.strip()

            elif searched_code:
                final_code = searched_code

            else:
                final_code = (
                    choice
                    if choice != "Other (Manual Entry)"
                    else manual_entry.strip()
                )

            if not final_code:
                st.error("Please select or enter an HTS code.")
                st.stop()

            @db_retry()
            def _save_product():
                with get_db_connection() as conn:
                    c = conn.cursor()

                    c.execute(
                        "SELECT product_id FROM product_classifications WHERE product_id = %s",
                        (product_id,),
                    )

                    if c.fetchone():
                        return False

                    c.execute(
                        '''
                        INSERT INTO product_classifications
                        (product_id, product_name, old_hs, new_hts, classified_by, classified_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ''',
                        (
                            product_id,
                            row['Product_Name'],
                            current_hs,
                            final_code,
                            st.session_state.username,
                            datetime.now().isoformat()
                        ),
                    )

                    return True

            saved = _save_product()

            get_classified_products.clear()

            if not saved:
                st.warning(
                    "⚡ A teammate just classified this item — skipping to the next one."
                )

            st.rerun()
# ===========================================================================
# STEP 2B – BATCH CLASSIFICATION
# ===========================================================================
elif step == "2B. Batch Classification":

    st.title("Step 2B: Batch Classification")

    approvals = get_category_approvals()
    cat_mappings = {k: v.get('approved_hts', '') for k, v in approvals.items()}

    classified_df = get_classified_products()
    classified_ids = (
        classified_df['product_id'].astype(str).tolist()
        if not classified_df.empty
        else []
    )

    remaining = products_df[
        ~products_df['Product_ID'].astype(str).isin(classified_ids)
    ].copy()

    if remaining.empty:
        st.success("🎉 No remaining products to classify.")
        st.stop()

    st.subheader("Find Products")

    search_term = st.text_input(
        "Search Product Name",
        placeholder="shirt, kurta, tunic, pants..."
    )

    category_options = sorted(
        remaining["Current_Odoo_HS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    category_filter = st.selectbox(
        "Filter by Odoo Category",
        ["All Categories"] + category_options
    )

    filtered = remaining.copy()

    if search_term:
        filtered = filtered[
            filtered["Product_Name"]
            .astype(str)
            .str.contains(search_term, case=False, na=False)
        ]

    if category_filter != "All Categories":
        filtered = filtered[
            filtered["Current_Odoo_HS"].astype(str)
            == category_filter
        ]

    st.write(f"Showing **{len(filtered)}** products")

    if filtered.empty:
        st.info("No products match the filters.")
        st.stop()

    filtered = filtered.copy()
    filtered["Select"] = False

    select_all = st.checkbox("Select All Visible Products")

    if select_all:
        filtered["Select"] = True

    editor_df = st.data_editor(
        filtered[
            [
                "Select",
                "Product_ID",
                "Product_Name",
                "Current_Odoo_HS"
            ]
        ],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="batch_editor"
    )

    selected = editor_df[editor_df["Select"]]

    st.write(f"Selected Products: **{len(selected)}**")

    if len(selected) == 0:
        st.stop()

    st.divider()

    st.subheader("Choose HTS Code")

    suggested_codes = []

    selected_categories = (
        filtered[
            filtered["Product_ID"]
            .isin(selected["Product_ID"])
        ]["Current_Odoo_HS"]
        .astype(str)
        .unique()
        .tolist()
    )

    for cat in selected_categories:

        approved = cat_mappings.get(cat, "")

        if approved:
            suggested_codes.extend([
                x.strip()
                for x in approved.split(",")
                if x.strip()
            ])

    suggested_codes = sorted(set(suggested_codes))

    if suggested_codes:

        selected_suggested = st.radio(
            "Suggested HTS Codes",
            ["None"] + suggested_codes,
            format_func=lambda x: (
                x
                if x == "None"
                else make_label(x)
            )
        )

    else:
        selected_suggested = "None"
        st.info("No suggested HTS codes available.")

    st.markdown("### Search All HTS Codes")

    all_hts_codes = sorted(desc_map.keys())

    searched_hts = st.selectbox(
        "Search HTS Database",
        options=[""] + all_hts_codes,
        format_func=lambda x: (
            "Search for HTS code..."
            if x == ""
            else make_label(x)
        )
    )

    manual_hts = st.text_input(
        "Custom HTS Code (optional)"
    )

    if st.button(
        f"Assign HTS To {len(selected)} Products",
        type="primary"
    ):

        final_code = None

        if manual_hts.strip():
            final_code = manual_hts.strip()

        elif searched_hts:
            final_code = searched_hts

        elif selected_suggested != "None":
            final_code = selected_suggested

        if not final_code:
            st.error("Please select or enter an HTS code.")
            st.stop()

        @db_retry()
        def _batch_save():

            with get_db_connection() as conn:

                c = conn.cursor()

                now = datetime.now().isoformat()

                inserted = 0

                for _, row in selected.iterrows():

                    product_id = str(row["Product_ID"])

                    c.execute(
                        """
                        SELECT product_id
                        FROM product_classifications
                        WHERE product_id = %s
                        """,
                        (product_id,)
                    )

                    if c.fetchone():
                        continue

                    c.execute(
                        """
                        INSERT INTO product_classifications
                        (
                            product_id,
                            product_name,
                            old_hs,
                            new_hts,
                            classified_by,
                            classified_at
                        )
                        VALUES (%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            product_id,
                            row["Product_Name"],
                            row["Current_Odoo_HS"],
                            final_code,
                            st.session_state.username,
                            now
                        )
                    )

                    inserted += 1

                return inserted

        inserted = _batch_save()

        get_classified_products.clear()

        st.success(
            f"✅ Assigned HTS {final_code} to {inserted} products."
        )

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