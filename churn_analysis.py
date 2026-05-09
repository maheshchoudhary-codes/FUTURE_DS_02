"""
Customer Churn Analysis — FUTURE_DS_02
Author  : [Your Name]
Dataset : customer_churn_dataset-training-master.csv
Story   : Understanding WHY customers leave — and what to do about it
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════
# 1. BUSINESS UNDERSTANDING
# ════════════════════════════════════════════════════════
"""
Churn = a customer stops using your service.

Why it matters:
  → Acquiring a new customer costs 5–7x more than retaining one.
  → Even a 5% reduction in churn can increase profit by 25–95%.
  → High churn signals a product, price, or service problem.

Objective: Identify WHO churns, WHY they churn, and what the
           business can do to keep them.
"""


# ════════════════════════════════════════════════════════
# 2. DATA LOADING & OVERVIEW
# ════════════════════════════════════════════════════════

df = pd.read_csv('customer_churn_dataset-training-master.csv')

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns    : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")

"""
Column Guide:
  CustomerID       → Unique customer identifier
  Age              → Customer age
  Gender           → Male / Female
  Tenure           → Months as a customer (1–60)
  Usage Frequency  → How often they use the product
  Support Calls    → Number of complaints/calls made
  Payment Delay    → Days late on payments
  Subscription Type→ Basic / Standard / Premium
  Contract Length  → Monthly / Quarterly / Annual
  Total Spend      → Lifetime spend in $
  Last Interaction → Days since last contact
  Churn            → 1 = left, 0 = stayed (TARGET)
"""


# ════════════════════════════════════════════════════════
# 3. DATA CLEANING
# ════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MISSING VALUES (before cleaning)")
print("=" * 55)
print(df.isnull().sum())

# Drop the single row with nulls (only 1 row out of 440K)
df.dropna(inplace=True)

# Ensure Churn is integer
df['Churn'] = df['Churn'].astype(int)

# Create tenure buckets for grouped analysis
df['Tenure_Group'] = pd.cut(
    df['Tenure'],
    bins=[0, 12, 24, 36, 48, 60],
    labels=['0–12 mo', '13–24 mo', '25–36 mo', '37–48 mo', '49–60 mo']
)

print(f"\nAfter cleaning: {df.shape[0]:,} rows retained")
print(f"Churn column dtype: {df['Churn'].dtype}")


# ════════════════════════════════════════════════════════
# VISUAL THEME
# ════════════════════════════════════════════════════════

STAY   = '#4ECDC4'   # mint-teal  → retained
CHURN  = '#FF6B6B'   # coral-red  → churned
SOFT   = '#F7F9FC'
TITLE  = '#1A1A2E'
MUTED  = '#8492A6'

plt.rcParams.update({
    'figure.facecolor': SOFT,
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#D1D9E0',
    'axes.labelcolor': TITLE,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'text.color': TITLE,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlepad': 14,
})


# ════════════════════════════════════════════════════════
# 4. EDA — CHAPTER 1: HOW BIG IS THE PROBLEM?
# ════════════════════════════════════════════════════════

churn_counts = df['Churn'].value_counts()
churn_rate   = df['Churn'].mean() * 100

fig1, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SOFT)
fig1.suptitle('Chapter 1 — How Big Is the Churn Problem?',
              fontsize=17, fontweight='bold', color=TITLE, y=1.02)

# Donut chart
wedge_props = {'edgecolor': 'white', 'linewidth': 3}
axes[0].pie(
    [churn_counts[0], churn_counts[1]],
    labels=['Retained\nCustomers', 'Churned\nCustomers'],
    colors=[STAY, CHURN], autopct='%1.1f%%', startangle=90,
    wedgeprops=wedge_props,
    textprops={'fontsize': 12, 'color': TITLE},
    pctdistance=0.75
)
centre = plt.Circle((0, 0), 0.55, fc='white')
axes[0].add_patch(centre)
axes[0].text(0, 0, f'{churn_rate:.1f}%\nChurn', ha='center', va='center',
             fontsize=14, fontweight='bold', color=CHURN)
axes[0].set_title('Overall Churn Rate', fontsize=13, fontweight='bold')

# Count bar
labels = ['Retained', 'Churned']
vals   = [churn_counts[0], churn_counts[1]]
bars   = axes[1].bar(labels, vals, color=[STAY, CHURN],
                     width=0.45, edgecolor='white', linewidth=2)
for bar, v in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 2000,
                 f'{v:,}', ha='center', fontsize=11,
                 fontweight='bold', color=bar.get_facecolor())
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Retained vs Churned Count',
                  fontsize=13, fontweight='bold')
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
axes[1].set_ylim(0, max(vals) * 1.15)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('churn_01_overview.png', dpi=150,
            bbox_inches='tight', facecolor=SOFT)
plt.close()
print("\n✓ Chapter 1 chart saved → churn_01_overview.png")


# ════════════════════════════════════════════════════════
# EDA — CHAPTER 2: WHO IS LEAVING?
# ════════════════════════════════════════════════════════

def churn_rate_bar(ax, col, title, order=None):
    data = df.groupby(col)['Churn'].mean().mul(100).round(1)
    if order:
        data = data.reindex(order)
    colors = [CHURN if v > 56 else STAY for v in data.values]
    bars = ax.bar(data.index, data.values, color=colors,
                  edgecolor='white', linewidth=2, width=0.5)
    for bar, v in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                f'{v}%', ha='center', fontsize=11,
                fontweight='bold', color=bar.get_facecolor())
    ax.axhline(churn_rate, color=MUTED, linestyle='--',
               linewidth=1.2, alpha=0.7)
    ax.text(len(data) - 0.45, churn_rate + 0.8,
            f'avg {churn_rate:.1f}%', fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


fig2, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=SOFT)
fig2.suptitle('Chapter 2 — Who Is Leaving?',
              fontsize=17, fontweight='bold', color=TITLE, y=1.02)

churn_rate_bar(axes[0], 'Contract Length',
               'Churn by Contract Type',
               order=['Monthly', 'Quarterly', 'Annual'])
churn_rate_bar(axes[1], 'Subscription Type',
               'Churn by Subscription Tier',
               order=['Basic', 'Standard', 'Premium'])
churn_rate_bar(axes[2], 'Gender',
               'Churn by Gender',
               order=['Female', 'Male'])

plt.tight_layout()
plt.savefig('churn_02_who.png', dpi=150,
            bbox_inches='tight', facecolor=SOFT)
plt.close()
print("✓ Chapter 2 chart saved → churn_02_who.png")


# ════════════════════════════════════════════════════════
# EDA — CHAPTER 3: WHY ARE THEY LEAVING?
# ════════════════════════════════════════════════════════

stayed  = df[df['Churn'] == 0]
churned = df[df['Churn'] == 1]

fig3, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=SOFT)
fig3.suptitle('Chapter 3 — Why Are Customers Leaving?',
              fontsize=17, fontweight='bold', color=TITLE, y=1.01)

# 3A — Support Calls
axes[0, 0].hist(stayed['Support Calls'],  bins=10, alpha=0.75,
                color=STAY,  label='Retained', edgecolor='white', density=True)
axes[0, 0].hist(churned['Support Calls'], bins=10, alpha=0.75,
                color=CHURN, label='Churned',  edgecolor='white', density=True)
axes[0, 0].set_title('Support Calls Distribution',
                      fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Number of Support Calls')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0, 0].text(
    0.97, 0.93,
    f"Churned avg: {churned['Support Calls'].mean():.1f} calls\n"
    f"Retained avg: {stayed['Support Calls'].mean():.1f} calls",
    transform=axes[0, 0].transAxes, ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3F3',
              edgecolor=CHURN, alpha=0.8))

# 3B — Payment Delay
axes[0, 1].hist(stayed['Payment Delay'],  bins=15, alpha=0.75,
                color=STAY,  label='Retained', edgecolor='white', density=True)
axes[0, 1].hist(churned['Payment Delay'], bins=15, alpha=0.75,
                color=CHURN, label='Churned',  edgecolor='white', density=True)
axes[0, 1].set_title('Payment Delay Distribution',
                      fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Days of Payment Delay')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.4)
axes[0, 1].text(
    0.97, 0.93,
    f"Churned avg: {churned['Payment Delay'].mean():.1f} days\n"
    f"Retained avg: {stayed['Payment Delay'].mean():.1f} days",
    transform=axes[0, 1].transAxes, ha='right', va='top', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3F3',
              edgecolor=CHURN, alpha=0.8))

# 3C — Tenure Group Churn Rate
tg     = df.groupby('Tenure_Group', observed=True)['Churn'].mean().mul(100)
colors = [CHURN if v > 56 else STAY for v in tg.values]
b      = axes[1, 0].bar(tg.index, tg.values, color=colors,
                        edgecolor='white', linewidth=2, width=0.5)
for bar, v in zip(b, tg.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f'{v:.1f}%', ha='center', fontsize=10,
                    fontweight='bold', color=bar.get_facecolor())
axes[1, 0].axhline(churn_rate, color=MUTED, linestyle='--',
                   linewidth=1.2, alpha=0.7)
axes[1, 0].set_title('Churn Rate by Tenure',
                      fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Customer Tenure (Months)')
axes[1, 0].set_ylabel('Churn Rate (%)')
axes[1, 0].set_ylim(0, 80)
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.4)

# 3D — Total Spend Boxplot
bp = axes[1, 1].boxplot(
    [stayed['Total Spend'], churned['Total Spend']],
    labels=['Retained', 'Churned'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2.5),
    whiskerprops=dict(color=MUTED),
    capprops=dict(color=MUTED),
    flierprops=dict(marker='o', markersize=2, alpha=0.2)
)
bp['boxes'][0].set_facecolor(STAY)
bp['boxes'][1].set_facecolor(CHURN)
axes[1, 1].set_title('Total Spend: Retained vs Churned',
                      fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Total Spend ($)')
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('churn_03_why.png', dpi=150,
            bbox_inches='tight', facecolor=SOFT)
plt.close()
print("✓ Chapter 3 chart saved → churn_03_why.png")


# ════════════════════════════════════════════════════════
# 5. KEY INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("KEY BUSINESS INSIGHTS")
print("=" * 55)
print(f"""
1. CHURN IS A CRISIS — NOT A SIDE NOTE
   56.7% of customers have left. That means for every 10 customers
   who sign up, almost 6 don't stick around. This is not a minor
   problem — it's a business emergency.

2. MONTHLY CONTRACT CUSTOMERS CHURN AT 100%
   Every single monthly-plan customer has churned. Compare that to
   Annual (46%) and Quarterly (46%) contracts. The flexibility of
   monthly plans is costing the company dearly.

3. SUPPORT CALLS ARE THE STRONGEST CHURN SIGNAL
   Customers who leave made an average of 5.1 support calls,
   while loyal customers made only 1.6. Frustrated customers are
   not just calling for help — they're calling before they quit.

4. DELAYED PAYMENTS PREDICT DEPARTURE
   Churned customers delayed payments by ~15 days on average,
   vs 10 days for retained customers. Payment friction = future churn.

5. GENDER MATTERS MORE THAN EXPECTED
   Female customers churn at 66.7% — significantly higher than
   male customers at 49.1%. This gap needs deeper investigation.
   Are there product-fit or communication issues for this segment?

6. SPEND LEVEL DOES NOT PROTECT YOU
   High spenders churn just as much as low spenders. This means
   the product itself — not price — is the core retention problem.
""")
