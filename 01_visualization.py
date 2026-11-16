"""
STEP 1 - VISUALIZATION
Explore the data with plots. Only the training set is used.

Run:  python 01_visualization.py
"""
from common import *


def main():
    header("STEP 1 - VISUALIZATION")
    train = X_train.assign(TenYearCHD=y_train)

    print("Shape of the full data:", df.shape)
    show(df.head(), "First rows")
    show(train.describe().T, "Summary statistics (train)")

    counts = y.value_counts().sort_index()
    print(f"\nTarget: {counts[0]} without CHD ({counts[0]/len(y):.1%}) | {counts[1]} with CHD ({counts[1]/len(y):.1%})")

    # 1) target distribution + CHD rate by age
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["No CHD (0)", "CHD (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 40, f"{v}\n({v/len(y):.1%})", ha="center")
    ax[0].set_ylim(0, counts.max() * 1.18)
    ax[0].set_title("Target distribution (all 4,238 people)")
    rate_by_age = train.groupby(pd.cut(train.age, [30, 40, 45, 50, 55, 60, 70]))[TARGET].mean()
    ax[1].bar(rate_by_age.index.astype(str), rate_by_age.values, color="#C44E52")
    ax[1].set_title("CHD rate by age group (train)")
    ax[1].set_ylabel("share with CHD")
    ax[1].tick_params(axis="x", rotation=30)
    savefig("01_target_and_age")

    # 2) continuous features split by target
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for a, c in zip(axes.ravel(), CONT):
        sns.kdeplot(data=train, x=c, hue=TARGET, common_norm=False, fill=True, alpha=.35, ax=a,
                    palette={0: "#4C72B0", 1: "#C44E52"})
        a.set_title(c)
        a.set_xlabel("")
    fig.suptitle("Continuous features: people with CHD (red) vs without (blue)", y=1.02)
    savefig("02_continuous_by_target")

    # 3) CHD rate inside each category
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for a, c in zip(axes.ravel(), BINARY + ORDINAL):
        r = train.groupby(c)[TARGET].mean()
        a.bar(r.index.astype(int).astype(str), r.values, color="#C44E52")
        a.axhline(y_train.mean(), ls="--", c="k", lw=1)
        a.set_title(c)
        a.set_ylabel("CHD rate")
    axes.ravel()[-1].axis("off")
    fig.suptitle("CHD rate inside each category (dashed line = overall rate)", y=1.02)
    savefig("03_categorical_by_target")

    # 4) correlations
    corr = train.corr()
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0], annot_kws={"size": 7}, cbar=False)
    ax[0].set_title("Correlation matrix (train)")
    ct = corr[TARGET].drop(TARGET).sort_values()
    ax[1].barh(ct.index, ct.values, color=np.where(ct.values > 0, "#C44E52", "#4C72B0"))
    ax[1].set_title("Correlation with TenYearCHD")
    savefig("04_correlation")

    show(ct.sort_values(ascending=False).to_frame("correlation with TenYearCHD"), "Correlation with the target")
    pairs = corr.drop(index=TARGET, columns=TARGET).where(np.triu(np.ones((15, 15), bool), 1)).stack()
    show(pairs[pairs.abs() > 0.5].sort_values(ascending=False).to_frame("correlation"),
         "Strongly related feature pairs (|r| > 0.5) -> redundancy, the reason for step 5")

    print("""
CONCLUSIONS
- The target is imbalanced: about 15% positive. Always answering "no CHD" is already ~85% accurate.
- Age and systolic blood pressure are the clearest signals.
- Every correlation with the target is weak (max ~0.23): accuracy far above the baseline is not expected.
- Several features duplicate each other (sysBP-diaBP, currentSmoker-cigsPerDay, glucose-diabetes).""")


if __name__ == "__main__":
    main()
