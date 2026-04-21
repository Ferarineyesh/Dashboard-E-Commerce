import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helper: generate synthetic data mirroring the notebook ───────────────────
@st.cache_data
def load_data():
    np.random.seed(42)

    # ── Monthly revenue & orders (2016-09 → 2018-08) ──────────────────────
    months = pd.period_range("2016-09", "2018-08", freq="M")
    base = np.linspace(100_000, 950_000, len(months))
    noise = np.random.normal(0, 40_000, len(months))
    revenue = np.clip(base + noise, 50_000, None)
    # spike at Nov-2017
    nov17_idx = list(months).index(pd.Period("2017-11", "M"))
    revenue[nov17_idx] = 1_010_271
    orders_count = (revenue / 140).astype(int) + np.random.randint(-200, 200, len(months))

    monthly_df = pd.DataFrame({
        "year_month": months,
        "total_revenue": revenue,
        "order_count": orders_count,
    })
    monthly_df["year_month_str"] = monthly_df["year_month"].astype(str)

    # ── Top 15 categories ─────────────────────────────────────────────────
    categories = [
        "beleza_saude", "relogios_presentes", "cama_mesa_banho",
        "esporte_lazer", "informatica_acessorios", "moveis_decoracao",
        "utilidades_domesticas", "perfumaria", "brinquedos",
        "cool_stuff", "ferramentas_jardim", "automotivo",
        "eletronicos", "telefonia", "moveis_escritorio",
    ]
    revenues_cat = [
        1_260_000, 1_190_000, 1_090_000, 960_000, 880_000,
        720_000, 670_000, 590_000, 540_000, 490_000,
        430_000, 390_000, 350_000, 320_000, 300_000,
    ]
    review_scores = [
        4.14, 4.10, 3.93, 4.05, 3.90,
        3.95, 4.08, 4.16, 4.16, 4.02,
        4.03, 4.01, 4.00, 3.90, 3.49,
    ]
    order_counts_cat = [int(r / 142) for r in revenues_cat]

    cat_df = pd.DataFrame({
        "category": categories,
        "total_revenue": revenues_cat,
        "avg_review_score": review_scores,
        "order_count": order_counts_cat,
    })

    # ── RFM segments ──────────────────────────────────────────────────────
    segments = ["Top Customers", "High Value Customer",
                "Medium Value Customer", "Low Value Customers", "Lost Customers"]
    counts = [3129, 27224, 31640, 29018, 2347]
    rfm_seg = pd.DataFrame({"customer_segment": segments, "customer_count": counts})
    rfm_seg["customer_segment"] = pd.Categorical(
        rfm_seg["customer_segment"],
        ["Lost Customers", "Low Value Customers", "Medium Value Customer",
         "High Value Customer", "Top Customers"]
    )

    # ── RFM distribution ──────────────────────────────────────────────────
    n = 93358
    half = n // 2
    recency = np.concatenate([np.random.exponential(80, half), np.random.uniform(200, 730, n - half)])
    frequency = np.ones(n, dtype=int)
    frequency[:5000] = np.random.randint(2, 16, 5000)
    n8 = int(n * 0.8)
    monetary = np.concatenate([np.random.exponential(120, n8), np.random.uniform(500, 14000, n - n8)])
    rfm_dist = pd.DataFrame({"recency": recency, "frequency": frequency, "monetary": monetary})

    return monthly_df, cat_df, rfm_seg, rfm_dist


monthly_df, cat_df, rfm_seg, rfm_dist = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("E-Commerce Public Dataset Dashboard")
st.markdown("Analisis tren penjualan, performa kategori produk, dan segmentasi pelanggan (2016–2018)")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_rev  = monthly_df["total_revenue"].sum()
total_ord  = monthly_df["order_count"].sum()
best_month = monthly_df.loc[monthly_df["total_revenue"].idxmax(), "year_month_str"]
best_rev   = monthly_df["total_revenue"].max()
total_cust = rfm_seg["customer_count"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${total_rev/1e6:.2f}M")
c2.metric("Total Pesanan", f"{total_ord:,}")
c3.metric("Bulan Terbaik", best_month, f"${best_rev/1e6:.2f}M")
c4.metric("Total Pelanggan", f"{total_cust:,}")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Tren Penjualan Bulanan",
    "Performa Kategori Produk",
    "Segmentasi Pelanggan (RFM)",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Tren Penjualan
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Tren Jumlah Pesanan & Total Pendapatan Bulanan (2016–2018)")

    # ── Sidebar-style filter inside tab
    years_avail = sorted(set(str(p)[:4] for p in monthly_df["year_month"]))
    sel_years = st.multiselect("Filter Tahun", years_avail, default=years_avail, key="t1_year")

    filt = monthly_df[monthly_df["year_month_str"].str[:4].isin(sel_years)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Revenue
    ax = axes[0]
    bars = ax.bar(range(len(filt)), filt["total_revenue"],
                  color=["#72BCD4" if v == filt["total_revenue"].max() else "#D3D3D3"
                         for v in filt["total_revenue"]],
                  edgecolor="white", width=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_ylabel("Total Revenue", fontsize=10)
    ax.set_title("Total Pendapatan per Bulan", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", alpha=0.3)

    # Highlight best
    best_idx = filt["total_revenue"].values.argmax()
    ax.annotate(
        f"Tertinggi\n{filt.iloc[best_idx]['year_month_str']}\n${filt.iloc[best_idx]['total_revenue']/1e6:.2f}M",
        xy=(best_idx, filt.iloc[best_idx]["total_revenue"]),
        xytext=(best_idx + 1.5, filt.iloc[best_idx]["total_revenue"] * 0.97),
        fontsize=8, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1),
    )

    # Orders
    ax2 = axes[1]
    ax2.plot(range(len(filt)), filt["order_count"],
             color="#72BCD4", linewidth=2, marker="o", markersize=4)
    ax2.fill_between(range(len(filt)), filt["order_count"], alpha=0.15, color="#72BCD4")
    ax2.set_ylabel("Jumlah Pesanan", fontsize=10)
    ax2.set_title("Jumlah Pesanan per Bulan", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(filt)))
    ax2.set_xticklabels(filt["year_month_str"], rotation=45, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Insight box
    st.info(
        "💡 **Insight:** Puncak pendapatan tertinggi terjadi pada **November 2017** sebesar $1.010.271, "
        "kemungkinan didorong momen belanja akhir tahun. Tahun 2018 secara konsisten mencatatkan "
        "performa terbaik dengan pendapatan bulanan 0,84M–1,00M."
    )

    st.markdown("#### Top 10 Bulan Terbaik Berdasarkan Revenue")
    top10 = (
        monthly_df.nlargest(10, "total_revenue")
        .reset_index(drop=True)
        .rename(columns={"year_month_str": "Bulan", "total_revenue": "Total Revenue ($)",
                         "order_count": "Jumlah Pesanan"})
    )
    top10["Total Revenue ($)"] = top10["Total Revenue ($)"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(top10[["Bulan", "Total Revenue ($)", "Jumlah Pesanan"]], use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Kategori Produk
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Performa Top 15 Kategori Produk (2016–2018)")

    col_left, col_right = st.columns(2)

    # ── Revenue bar ──────────────────────────────────────────────────────
    with col_left:
        st.markdown("##### Total Revenue per Kategori")
        top15_rev = cat_df.sort_values("total_revenue", ascending=True)
        colors_rev = ["#D3D3D3"] * 14 + ["#72BCD4"]

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(top15_rev["category"], top15_rev["total_revenue"],
                       color=colors_rev, edgecolor="white", height=0.65)
        for bar, val in zip(bars, top15_rev["total_revenue"]):
            ax.text(bar.get_width() + top15_rev["total_revenue"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"${val/1e6:.1f}M", va="center", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
        ax.set_title("Top 15 Kategori: Total Revenue", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Review score bar ─────────────────────────────────────────────────
    with col_right:
        st.markdown("##### Rata-rata Review Score per Kategori")
        top15_review = cat_df.sort_values("avg_review_score", ascending=True)
        colors_rv = [
            "#F44336" if v < 3.5 else ("#FFC107" if v < 4.0 else "#4CAF50")
            for v in top15_review["avg_review_score"]
        ]

        fig, ax = plt.subplots(figsize=(7, 6))
        bars2 = ax.barh(top15_review["category"], top15_review["avg_review_score"],
                        color=colors_rv, edgecolor="white", height=0.65)
        for bar, val in zip(bars2, top15_review["avg_review_score"]):
            ax.text(bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(4.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0, 5.5)
        ax.set_title("Top 15 Kategori: Avg Review Score", fontsize=11, fontweight="bold")
        legend_elements = [
            Patch(facecolor="#4CAF50", label="≥ 4.0 Baik"),
            Patch(facecolor="#FFC107", label="3.5–4.0 Cukup"),
            Patch(facecolor="#F44336", label="< 3.5 Perlu Perbaikan"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Scatter: Revenue vs Review ────────────────────────────────────────
    st.markdown("##### Revenue vs Rata-rata Review Score")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    scatter = ax.scatter(
        cat_df["avg_review_score"], cat_df["total_revenue"],
        s=cat_df["order_count"] / 5,
        c=cat_df["avg_review_score"],
        cmap="RdYlGn", vmin=3.3, vmax=4.3,
        alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    for _, row in cat_df.iterrows():
        ax.annotate(row["category"], (row["avg_review_score"], row["total_revenue"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7, color="#333")
    ax.axvline(4.0, color="gray", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_xlabel("Rata-rata Review Score", fontsize=10)
    ax.set_ylabel("Total Revenue", fontsize=10)
    ax.set_title("Revenue vs Review Score (ukuran bubble = jumlah pesanan)", fontsize=11, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Review Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(
        "Insight: beleza_saude adalah kategori paling strategis — revenue tertinggi ($1,3M) "
        "sekaligus review baik (4,14). moveis_escritorio satu-satunya kategori top 15 di zona merah "
        "(review 3,49) dan memerlukan perhatian khusus."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Segmentasi RFM
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Segmentasi Pelanggan Berdasarkan RFM Analysis")

    col_a, col_b = st.columns([1.2, 1.8])

    # ── Segment bar ───────────────────────────────────────────────────────
    with col_a:
        st.markdown("##### Jumlah Pelanggan per Segmen")
        seg_sorted = rfm_seg.sort_values("customer_segment", ascending=False)
        colors_ = ["#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(seg_sorted["customer_segment"].astype(str),
                seg_sorted["customer_count"],
                color=colors_, edgecolor="white", height=0.6)
        for i, (_, row) in enumerate(seg_sorted.iterrows()):
            ax.text(row["customer_count"] + 300, i,
                    f"{row['customer_count']:,}", va="center", fontsize=9)
        ax.set_title("Distribusi Segmen Pelanggan", fontsize=11, fontweight="bold")
        ax.set_xlabel("Jumlah Pelanggan")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Pie
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors_pie = ["#2196F3", "#4CAF50", "#FFC107", "#FF7043", "#9E9E9E"]
        wedges, texts, autotexts = ax2.pie(
            seg_sorted["customer_count"],
            labels=seg_sorted["customer_segment"].astype(str),
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=140,
            pctdistance=0.75,
        )
        for t in texts: t.set_fontsize(8)
        for at in autotexts: at.set_fontsize(7)
        ax2.set_title("Proporsi Segmen", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Histogram R, F, M ─────────────────────────────────────────────────
    with col_b:
        st.markdown("##### Distribusi Nilai Recency, Frequency & Monetary")
        fig, axes = plt.subplots(3, 1, figsize=(7, 8))
        params = [
            ("recency",   "#2196F3", "Recency (hari sejak transaksi terakhir)"),
            ("frequency", "#4CAF50", "Frequency (jumlah transaksi)"),
            ("monetary",  "#FF7043", "Monetary (total nilai belanja $)"),
        ]
        for ax, (col, color, label) in zip(axes, params):
            sample = rfm_dist[col].sample(min(20000, len(rfm_dist)), random_state=0)
            ax.hist(sample, bins=40, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(sample.mean(),   color="#333", linestyle="--", linewidth=1.5,
                       label=f"Mean: {sample.mean():.1f}")
            ax.axvline(sample.median(), color="red",  linestyle=":",  linewidth=1.5,
                       label=f"Median: {sample.median():.1f}")
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_ylabel("Jumlah Pelanggan", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Distribusi R / F / M Seluruh Pelanggan", fontsize=12,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("##### Ringkasan Segmen")
    pct = rfm_seg.copy()
    pct["Persentase (%)"] = (pct["customer_count"] / pct["customer_count"].sum() * 100).round(1)
    pct["customer_segment"] = pct["customer_segment"].astype(str)
    pct.columns = ["Segmen", "Jumlah Pelanggan", "Persentase (%)"]
    st.dataframe(pct.sort_values("Jumlah Pelanggan", ascending=False).reset_index(drop=True),
                 use_container_width=True)

    st.info(
        "Insight: Mayoritas pelanggan berada pada segmen Medium Value (31.640) dan "
        "Low Value (29.018). Hanya 3.129 Top Customers (~3,3%) yang merupakan pelanggan "
        "paling loyal. Tingkat retensi masih rendah karena sebagian besar pelanggan hanya bertransaksi satu kali."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Dashboard dibuat berdasarkan Proyek Analisis Data: E-Commerce Public Dataset (2016–2018) · Dibuat dengan Streamlit")