"""Figure generation for SAFIRA-SSA."""

from __future__ import annotations

import os
import warnings
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import DEFAULT_FIGURES_DIR


class SAIPlotter:
    """Generate the figures used in the SAI forecasting analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        figures_dir: str | Path = DEFAULT_FIGURES_DIR,
        show: bool = False,
        world_file: str | None = None,
    ):
        self.df = df.copy()
        self.figures_dir = Path(figures_dir)
        self.show = bool(show)
        self.world_file = world_file
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        if "year" in self.df.columns:
            self.latest_year = int(self.df["year"].max())
        else:
            raise ValueError("Plotting dataframe must contain a 'year' column.")

        self.latest = self.df[self.df.year == self.latest_year].copy()
        bloc_map = {
            "AGO": "SADC",
            "BWA": "SADC",
            "NAM": "SADC",
            "ZAF": "SADC",
            "ZMB": "SADC",
            "ZWE": "SADC",
            "MOZ": "SADC",
            "SWZ": "SADC",
            "LSO": "SADC",
            "BEN": "ECOWAS",
            "BFA": "ECOWAS",
            "CIV": "ECOWAS",
            "CPV": "ECOWAS",
            "GMB": "ECOWAS",
            "GHA": "ECOWAS",
            "GIN": "ECOWAS",
            "GNB": "ECOWAS",
            "NER": "ECOWAS",
            "NGA": "ECOWAS",
            "SEN": "ECOWAS",
            "SLE": "ECOWAS",
            "TGO": "ECOWAS",
            "BDI": "EAC",
            "KEN": "EAC",
            "RWA": "EAC",
            "TZA": "EAC",
            "UGA": "EAC",
        }
        self.latest["bloc"] = self.latest.country_code.map(bloc_map).fillna("OTHER")

    def _plot_libs(self):
        import matplotlib

        if not self.show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use("ggplot")
        sns.set_context("notebook", font_scale=1.05)
        warnings.filterwarnings("ignore")
        return plt, sns

    def _path(self, filename: str) -> str:
        return str(self.figures_dir / filename)

    def _finish(self, plt, filename: str, dpi: int = 300) -> str:
        path = self._path(filename)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        if self.show:
            plt.show()
        plt.close()
        return path

    def _load_world(self):
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "The choropleth plot requires geopandas and mapclassify. "
                "Install SAFIRA-SSA with the plot extra."
            ) from exc

        source = self.world_file or (
            "https://naturalearth.s3.amazonaws.com/"
            "110m_cultural/ne_110m_admin_0_countries.zip"
        )
        gdf = gpd.read_file(source)
        if "ISO_A3" in gdf.columns:
            gdf = gdf.rename(columns={"ISO_A3": "iso3"})
        elif "iso_a3" in gdf.columns:
            gdf = gdf.rename(columns={"iso_a3": "iso3"})
        else:
            raise KeyError("ISO-3 field not found in Natural Earth data.")
        return gdf[["iso3", "geometry"]]

    def plot_choropleth(self) -> str:
        """Map mean SAI from 2020 through the latest available year."""
        plt, _ = self._plot_libs()
        mean20 = (
            self.df[self.df.year >= 2020]
            .groupby("country_code")["SAI"]
            .mean()
            .reset_index()
            .rename(columns={"country_code": "iso3"})
        )
        world = self._load_world()
        geo = world.merge(mean20, on="iso3", how="right")
        ax = geo.plot(
            column="SAI",
            cmap="RdYlGn",
            scheme="NaturalBreaks",
            legend=True,
            edgecolor="gray",
            linewidth=0.5,
            figsize=(8, 6),
        )
        ax.set_axis_off()
        ax.set_title(f"Skills Advancement Index - mean 2020-{self.latest_year}", fontsize=14)
        return self._finish(plt, "plot_choropleth.png")

    def plot_spaghetti(self) -> str:
        """Plot all country SAI trajectories with the SSA median."""
        plt, _ = self._plot_libs()
        plt.figure(figsize=(10, 6))
        for _, group in self.df.groupby("country_code"):
            plt.plot(group.year, group.SAI, color="lightgray", alpha=0.6, linewidth=1)
        median = self.df.groupby("year")["SAI"].median()
        plt.plot(median.index, median.values, color="black", linewidth=2.5, label="SSA median")
        plt.xlabel("Year")
        plt.ylabel("SAI (0-100)")
        plt.title(f"SSA SAI trajectories 2000-{self.latest_year}")
        plt.legend()
        return self._finish(plt, "plot_spaghetti.png")

    def plot_pillar_heatmap(self, eps: float = 1e-6, min_valid_pillars: int = 2) -> str | None:
        """Cluster countries by their latest four-pillar SAI profiles."""
        plt, sns = self._plot_libs()
        pillars = ["dim_foundational", "dim_advanced", "dim_digital", "dim_labor"]
        frame = self.df[self.df["year"] == self.latest_year].copy()
        if frame.empty:
            print(f"[WARN] No rows for year {self.latest_year}.")
            return None

        mat = frame.set_index("country_code")[pillars].apply(pd.to_numeric, errors="coerce")
        mat = mat[mat.notna().sum(axis=1) >= min_valid_pillars]
        if mat.shape[0] < 2:
            print(f"[WARN] Too few countries with at least {min_valid_pillars} pillars.")
            return None

        mat = mat.fillna(mat.median(axis=0, skipna=True))
        z = mat.apply(lambda col: (col - col.mean()) / (col.std() + eps))
        z = z.loc[:, z.std(axis=0, skipna=True) > 1e-12]
        if z.shape[1] < 2 or z.shape[0] < 2:
            plt.figure(figsize=(7, 10))
            sns.heatmap(z, cmap="coolwarm", cbar=True)
            plt.title(f"Pillar profile (z-scores, {self.latest_year})")
            return self._finish(plt, "plot_pillar_heatmap.png")

        cluster = sns.clustermap(
            z,
            cmap="coolwarm",
            figsize=(7, 10),
            col_cluster=False,
            dendrogram_ratio=0.15,
        )
        cluster.fig.suptitle(f"Pillar profile (z-scores, {self.latest_year})", y=1.02)
        path = self._path("plot_pillar_heatmap.png")
        cluster.fig.savefig(path, dpi=300, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(cluster.fig)
        return path

    def plot_box(self) -> str:
        """Compare latest SAI distributions by regional bloc."""
        plt, sns = self._plot_libs()
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="bloc", y="SAI", data=self.latest, palette="Set3")
        plt.ylabel(f"SAI ({self.latest_year})")
        plt.xlabel("")
        plt.title("Distribution of SAI by regional bloc")
        return self._finish(plt, "plot_box.png")

    @staticmethod
    def _resolve_col(df: pd.DataFrame, preferred: str, variants: Sequence[str]) -> str | None:
        if preferred in df.columns:
            return preferred
        for variant in variants:
            if variant in df.columns:
                return variant

        import re
        import unicodedata

        def norm(value: str) -> str:
            text = unicodedata.normalize("NFKD", str(value))
            text = text.encode("ascii", "ignore").decode("ascii")
            return re.sub(r"[^A-Za-z0-9]+", "", text).lower()

        target = norm(preferred)
        mapping = {column: norm(column) for column in df.columns}
        for column, normalized in mapping.items():
            if normalized == target:
                return column
        for column, normalized in mapping.items():
            if target in normalized:
                return column
        return None

    def plot_gdp_scatter(
        self,
        year: int | None = None,
        highlight_codes: tuple[str, ...] = ("NGA",),
        annotate: bool = True,
        annotate_top_n: int = 0,
        filename: str | None = None,
    ) -> str | None:
        """Scatter GDP per capita against SAI for one year."""
        plt, sns = self._plot_libs()
        show_year = self.latest_year if year is None else int(year)
        frame = self.df[self.df["year"] == show_year].copy()
        if frame.empty:
            print(f"[WARN] No data for year {show_year}.")
            return None

        gdp_col = self._resolve_col(
            frame,
            "GDP_per_Capita",
            ["GDP_Per_Capita", "GDP_per_Capita_USD", "GDP per Capita", "GDP per capita (current US$)"],
        )
        gov_col = self._resolve_col(
            frame,
            "Gov_Effectiveness",
            ["Government_Effectiveness", "Govt_Effectiveness", "GovernmentEffectiveness"],
        )
        missing = [name for name, col in [("GDP_per_Capita", gdp_col), ("Gov_Effectiveness", gov_col)] if col is None]
        if missing:
            print(f"[WARN] Could not find: {', '.join(missing)}.")
            return None

        g = frame[["country_code", "SAI", gdp_col, gov_col]].copy()
        g[gdp_col] = pd.to_numeric(g[gdp_col], errors="coerce")
        g = g[g[gdp_col] > 0]
        if g.empty:
            print(f"[WARN] Year {show_year}: no positive GDP per capita values.")
            return None

        g["log_GDPpc"] = np.log10(g[gdp_col].values)
        ge = pd.to_numeric(g[gov_col], errors="coerce")
        ge_imputed = ge.fillna(ge.median() if ge.notna().any() else 0.0)
        try:
            g["GovQ"] = pd.qcut(ge_imputed, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            g["GovQ"] = pd.cut(ge_imputed, bins=4, labels=["Q1", "Q2", "Q3", "Q4"])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=g,
            x="log_GDPpc",
            y="SAI",
            hue="GovQ",
            palette="viridis",
            alpha=0.85,
            edgecolor="k",
            s=70,
        )
        if highlight_codes:
            hi = g[g["country_code"].isin(highlight_codes)].copy()
            if not hi.empty:
                sns.scatterplot(
                    data=hi,
                    x="log_GDPpc",
                    y="SAI",
                    color="tab:orange",
                    edgecolor="k",
                    s=160,
                    marker="o",
                    label="Highlighted",
                )
                if annotate:
                    for _, row in hi.iterrows():
                        plt.text(
                            row["log_GDPpc"] + 0.02,
                            row["SAI"],
                            row["country_code"],
                            va="center",
                            ha="left",
                            fontsize=9,
                            weight="bold",
                        )

        if annotate_top_n and annotate_top_n > 0:
            top = g.nlargest(annotate_top_n, "SAI")
            for _, row in top.iterrows():
                plt.text(row["log_GDPpc"] + 0.02, row["SAI"], row["country_code"], va="center", ha="left", fontsize=8)

        plt.xlabel("log10 GDP per capita (US$)")
        plt.ylabel(f"SAI ({show_year})")
        plt.title(f"Economic capacity vs. skills advancement - {show_year}")
        out_name = filename or f"plot_gdp_scatter_{show_year}.png"
        return self._finish(plt, out_name)

    def plot_corr(self, lower_triangle: bool = False) -> str:
        """Plot correlations among SAI pillars and contextual variables."""
        plt, sns = self._plot_libs()
        corr_vars = [
            "SAI",
            "dim_foundational",
            "dim_advanced",
            "dim_digital",
            "dim_labor",
            "GDP_Growth",
            "R_and_D_Expenditure",
            "Electric_Power_Consumption",
            "Gov_Effectiveness",
            "Youth_Unemployment",
        ]
        existing = [column for column in corr_vars if column in self.df.columns]
        corr = self.df[existing].corr()
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool)) if lower_triangle else None
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=lower_triangle,
            cbar_kws={"shrink": 0.75} if lower_triangle else None,
        )
        title = "Pair-wise correlations (lower triangle)" if lower_triangle else "Pair-wise correlations"
        plt.title(title)
        return self._finish(plt, "plot_corr_lower.png" if lower_triangle else "plot_corr.png")

    def plot_slope(self, base_year: int = 2000) -> str | None:
        """Show country-level SAI change between a base year and the latest year."""
        plt, _ = self._plot_libs()
        if base_year not in self.df["year"].unique():
            print(f"[WARN] Base year {base_year} is not present in the panel.")
            return None

        base = self.df[self.df.year == base_year][["country_code", "SAI"]].set_index("country_code")
        end = self.df[self.df.year == self.latest_year][["country_code", "SAI"]].set_index("country_code")
        slope = base.join(end, lsuffix=f"_{base_year}", rsuffix=f"_{self.latest_year}").dropna().sort_values(f"SAI_{base_year}")
        if slope.empty:
            print("[WARN] No overlapping countries for slope plot.")
            return None

        codes = slope.index.to_list()
        y0 = slope[f"SAI_{base_year}"].values
        y1 = slope[f"SAI_{self.latest_year}"].values
        fig, ax = plt.subplots(figsize=(9, 10))
        for start, finish in zip(y0, y1):
            ax.plot([0, 1], [start, finish], color="tab:blue" if finish > start else "tab:red", linewidth=1)

        y0_labels = np.linspace(y0.min(), y0.max(), len(y0))
        y1_labels = np.linspace(y1.min(), y1.max(), len(y1))
        order0, order1 = np.argsort(y0), np.argsort(y1)
        y0_spaced, y1_spaced = np.empty_like(y0_labels), np.empty_like(y1_labels)
        y0_spaced[order0] = y0_labels
        y1_spaced[order1] = y1_labels

        for code, true_y, label_y in zip(codes, y0, y0_spaced):
            ax.plot([0, -0.05], [true_y, label_y], color="gray", linewidth=0.5, alpha=0.7)
            ax.text(-0.07, label_y, code, ha="right", va="center", fontsize=8)
        for code, true_y, label_y in zip(codes, y1, y1_spaced):
            ax.plot([1, 1.05], [true_y, label_y], color="gray", linewidth=0.5, alpha=0.7)
            ax.text(1.07, label_y, code, ha="left", va="center", fontsize=8)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([str(base_year), str(self.latest_year)])
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylabel("SAI")
        ax.set_title(f"Change in SAI, {base_year} to {self.latest_year}")
        return self._finish(plt, "plot_slope.png")

    @staticmethod
    def _radar(ax, values, color, label):
        angles = np.linspace(0, 2 * np.pi, 5)[:-1]
        values = np.r_[values, values[0]]
        angles = np.r_[angles, angles[0]]
        ax.plot(angles, values, color=color, label=label, linewidth=2)
        ax.fill(angles, values, alpha=0.25, color=color)

    def plot_radar(self) -> str | None:
        """Plot four-pillar profiles for representative countries."""
        plt, _ = self._plot_libs()
        pillars = ["dim_foundational", "dim_advanced", "dim_digital", "dim_labor"]
        archetypes = {"Front-runner": "MUS", "Median": "GHA", "Laggard": "NER"}
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        colors = ["tab:green", "tab:blue", "tab:red"]
        plotted = 0
        for (label, code), color in zip(archetypes.items(), colors):
            rows = self.latest[self.latest.country_code == code]
            if rows.empty:
                continue
            values = rows[pillars].values.squeeze() / 100
            self._radar(ax, values, color=color, label=label)
            plotted += 1
        if plotted == 0:
            print("[WARN] Representative countries for radar plot are not present.")
            plt.close(fig)
            return None
        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(["Found.", "Adv.", "Digital", "Labor"])
        ax.set_yticklabels([])
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title(f"Four-pillar profiles ({self.latest_year})")
        return self._finish(plt, "plot_radar.png")

    def plot_umap(
        self,
        year: int | None = None,
        min_features: int = 5,
        force_features: list[str] | None = None,
    ) -> str | None:
        """Embed countries into a two-dimensional feature space with UMAP."""
        try:
            from sklearn.preprocessing import StandardScaler
            from umap import UMAP
        except ImportError as exc:
            print(f"[WARN] UMAP plot skipped: missing dependency ({exc}).")
            return None

        import re
        import unicodedata

        plt, sns = self._plot_libs()
        use_year = self.latest_year if year is None else int(year)
        frame = self.df[self.df["year"] == use_year].copy()
        if frame.empty:
            print(f"[WARN] No data for year {use_year}.")
            return None

        embed_vars = [
            "High_Tech_Exports_USD",
            "Public_Ed_Expenditure",
            "Labor_Force_Participation",
            "Population_Growth",
            "GDP_Per_Capita",
            "Electric_Power_Consumption",
            "Life_Expectancy_Birth",
            "Gov_Effectiveness",
            "Rural_Urban_Divide_Proxy",
            "Inflation_Consumer_Prices",
        ]
        candidates = force_features or embed_vars

        def norm(value: str) -> str:
            text = unicodedata.normalize("NFKD", str(value))
            text = text.encode("ascii", "ignore").decode("ascii")
            return re.sub(r"[^A-Za-z0-9]+", "", text).lower()

        norm_map = {norm(column): column for column in frame.columns}
        resolved, missing = [], []
        for feature in candidates:
            normalized = norm(feature)
            if normalized in norm_map:
                resolved.append(norm_map[normalized])
                continue
            hits = [column for key, column in norm_map.items() if normalized in key]
            if hits:
                resolved.append(hits[0])
            else:
                missing.append(feature)

        seen = set()
        resolved = [column for column in resolved if not (column in seen or seen.add(column))]
        if len(resolved) < min_features:
            print(f"[WARN] Not enough usable features for UMAP. Resolved {resolved}; missing {missing}.")
            return None

        xnum = frame[resolved].apply(pd.to_numeric, errors="coerce")
        medians = xnum.median(axis=0, skipna=True)
        xfilled = xnum.fillna(medians)
        std = xfilled.std(axis=0, skipna=True)
        good_cols = [column for column in resolved if pd.notna(medians.get(column)) and float(std.get(column, 0.0)) > 1e-12]
        if len(good_cols) < 2:
            print("[WARN] Too few informative features for UMAP after cleaning.")
            return None

        scaled = StandardScaler().fit_transform(xfilled[good_cols].to_numpy(dtype=float))
        n_samples = scaled.shape[0]
        if n_samples < 3:
            print(f"[WARN] Too few countries to embed for {use_year} (n={n_samples}).")
            return None
        embedding = UMAP(n_neighbors=max(2, min(15, n_samples - 1)), random_state=42).fit_transform(scaled)

        sai = pd.to_numeric(frame["SAI"], errors="coerce")
        hue = None
        if sai.notna().sum() > 0:
            try:
                hue = pd.qcut(sai.fillna(sai.median()), 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            except ValueError:
                hue = pd.cut(sai.fillna(sai.median()), bins=4, labels=["Q1", "Q2", "Q3", "Q4"])

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=hue,
            palette="viridis" if hue is not None else None,
            s=100,
            edgecolor="k",
            alpha=0.9,
        )
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title(f"Latent country clusters - {use_year}" + (" (colour = SAI quartile)" if hue is not None else ""))
        if hue is not None:
            plt.legend(title="SAI quartile")
        else:
            plt.legend([], [], frameon=False)
        return self._finish(plt, f"plot_umap_{use_year}.png")

    def plot_forecast_diag(
        self,
        country_code: str = "NGA",
        weights_file: str = "models/time_series_sai_model_weights.pth",
        assets_file: str = "models/time_series_sai_assets.pkl",
    ) -> str:
        """Plot one-step-ahead diagnostics for a country using saved model assets."""
        import torch

        from .forecast import load_plot_model

        plt, _ = self._plot_libs()
        model, assets = load_plot_model(weights_file, assets_file, device="cpu")
        feats = assets["input_features"]
        if feats != ["SAI_scaled"]:
            raise ValueError("This diagnostic expects input_features == ['SAI_scaled'].")

        lookback = int(assets["lookback"])
        target_scaler = assets["target_scaler"]
        series = self.df[self.df.country_code == country_code].sort_values("year")
        if series.shape[0] < lookback + 1:
            raise ValueError(f"Not enough observations for {country_code}.")

        y = series["SAI"].values.astype(float).reshape(-1, 1)
        y_scaled = target_scaler.transform(y).ravel().astype(np.float32)
        X = y_scaled.reshape(-1, 1)
        preds_scaled = []
        for i in range(lookback, len(X)):
            window = torch.tensor(X[i - lookback : i, :]).unsqueeze(0)
            with torch.no_grad():
                preds_scaled.append(float(model(window).item()))

        years_pred = series.year.iloc[lookback:]
        preds = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
        plt.figure(figsize=(8, 4))
        plt.plot(series.year, series.SAI, label="Actual")
        plt.plot(years_pred, preds, label="Forecast", linestyle="--", linewidth=1.8)
        plt.xlabel("Year")
        plt.ylabel("SAI (0-100)")
        plt.title(f"{country_code}: forecast diagnostic")
        plt.legend()
        return self._finish(plt, f"plot_forecast_diag_{country_code}.png")

    def plot_nigeria_comparison(self, selected_codes: Sequence[str] = ("RWA", "UGA", "KEN", "GHA")) -> str:
        """Compare Nigeria with selected peer countries."""
        plt, _ = self._plot_libs()
        peers = self.latest[self.latest.country_code.isin(list(selected_codes) + ["NGA"])].sort_values("SAI")
        colors = ["tab:orange" if code == "NGA" else "tab:blue" for code in peers.country_code]
        plt.figure(figsize=(8, 4))
        plt.barh(peers.country_code, peers.SAI, color=colors)
        for _, row in peers.iterrows():
            plt.text(row.SAI + 0.5, row.country_code, f"{row.SAI:,.1f}", va="center")
        plt.xlabel(f"SAI ({self.latest_year})")
        plt.title("Nigeria vs peer countries")
        return self._finish(plt, "plot_nigeria_comparison.png")

    def plot_nigeria_comparison_seaborn(self, selected_codes: Sequence[str] = ("RWA", "UGA", "KEN", "GHA")) -> str:
        """Render the Nigeria comparison with Seaborn styling."""
        plt, sns = self._plot_libs()
        comp_codes = ["NGA"] + list(selected_codes)
        data = self.latest[self.latest.country_code.isin(comp_codes)].copy()
        if data.empty:
            raise ValueError("Could not find data for the specified countries.")
        order = data.sort_values("SAI", ascending=True)
        plt.figure(figsize=(6, 4))
        sns.barplot(data=order, y="country_code", x="SAI", palette="viridis")
        plt.xlabel(f"SAI ({self.latest_year})")
        plt.ylabel("")
        plt.title("How Nigeria's SAI compares with peers")
        return self._finish(plt, "plot_nigeria_comparison_with_seaborn.png")

    def plot_spaghetti_selected(
        self,
        country_codes: Sequence[str] = ("KEN", "RWA", "GHA", "ZAF", "NGA", "COD"),
        filename: str = "plot_spaghetti_selected.png",
    ) -> str:
        """Highlight selected countries against the full SSA trajectory background."""
        plt, sns = self._plot_libs()
        plt.figure(figsize=(10, 6))
        for _, group in self.df.groupby("country_code"):
            plt.plot(group.year, group.SAI, color="lightgray", alpha=0.4, linewidth=1)
        palette = sns.color_palette("deep", len(country_codes))
        for code, color in zip(country_codes, palette):
            group = self.df[self.df.country_code == code]
            if not group.empty:
                plt.plot(group.year, group.SAI, label=code, color=color, linewidth=2.5)
        median = self.df.groupby("year")["SAI"].median()
        plt.plot(median.index, median.values, color="black", linewidth=2.5, label="SSA median")
        plt.xlabel("Year")
        plt.ylabel("SAI (0-100)")
        plt.title(f"Selected SSA SAI trajectories (2000-{self.latest_year})")
        plt.legend(ncol=2)
        return self._finish(plt, filename)

    def plot_future_skills_predictions(
        self,
        horizon: int = 10,
        weights_file: str = "models/time_series_sai_model_weights.pth",
        assets_file: str = "models/time_series_sai_assets.pkl",
    ) -> tuple[pd.DataFrame, str]:
        """Forecast the SSA median SAI horizon using saved LSTM assets."""
        import torch

        from .forecast import build_feature_matrix_for_ssa_median, load_plot_model

        plt, _ = self._plot_libs()
        model, assets = load_plot_model(weights_file=weights_file, assets_file=assets_file, device="cpu")
        target_scaler = assets["target_scaler"]
        exog_scalers = assets["exog_scalers"]
        input_features = assets["input_features"]
        lookback = int(assets["lookback"])

        years = np.sort(self.df["year"].unique())
        X_all = build_feature_matrix_for_ssa_median(
            self.df, years, input_features, target_scaler, exog_scalers
        )
        if X_all.shape[0] < lookback + 1:
            raise ValueError(f"SSA median series too short for lookback={lookback}.")

        preds_scaled = []
        model.eval()
        with torch.no_grad():
            for i in range(lookback, X_all.shape[0]):
                window = torch.tensor(X_all[i - lookback : i, :]).float().unsqueeze(0)
                preds_scaled.append(model(window).item())

        preds_insample = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
        years_insample = years[lookback:]
        cur_seq = torch.tensor(X_all[-lookback:, :]).float().unsqueeze(0)
        last_exog = [None if feature == "SAI_scaled" else float(cur_seq[0, -1, j].item()) for j, feature in enumerate(input_features)]

        future_scaled = []
        with torch.no_grad():
            for _ in range(horizon):
                y_hat = float(model(cur_seq).item())
                next_vec = [y_hat if feature == "SAI_scaled" else last_exog[j] for j, feature in enumerate(input_features)]
                next_vec = torch.tensor(next_vec).float().view(1, 1, -1)
                cur_seq = torch.cat([cur_seq[:, 1:, :], next_vec], dim=1)
                future_scaled.append(y_hat)

        future_unscaled = target_scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()
        future_years = np.arange(years.max() + 1, years.max() + 1 + horizon)
        ssa_median_actual = self.df.groupby("year")["SAI"].median().reindex(years)

        plt.figure(figsize=(9, 4))
        plt.plot(years, ssa_median_actual.values, label="Actual median", color="tab:blue")
        plt.plot(years_insample, preds_insample, label="Prediction", linestyle="--", linewidth=1.8, color="tab:green")
        plt.plot(future_years, future_unscaled, label="Forecast", color="tab:orange", linewidth=2.5)
        plt.xlabel("Year")
        plt.ylabel("SAI (0-100)")
        plt.title(f"SSA median SAI - Forecast ({horizon}-year horizon)")
        plt.legend()
        path = self._finish(plt, "plot_future_skills_projection.png")
        return pd.DataFrame({"year": future_years, "SAI_predicted": future_unscaled}), path

    def zip_figures(self, zip_name: str = "figures.zip") -> str:
        """Archive generated figure files."""
        zip_path = Path(zip_name)
        if not zip_path.is_absolute() and zip_path.parent == Path("."):
            zip_path = self.figures_dir.parent / zip_name
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.figures_dir):
                for filename in files:
                    path = Path(root) / filename
                    zf.write(path, path.relative_to(self.figures_dir.parent))
        print(f"[INFO] Created archive {zip_path}")
        return str(zip_path)


def run_named_plots(
    plotter: SAIPlotter,
    names: Iterable[str],
    weights_file: str = "models/time_series_sai_model_weights.pth",
    assets_file: str = "models/time_series_sai_assets.pkl",
    forecast_diag_countries: Sequence[str] = ("NGA", "CIV", "KEN", "CMR", "RWA", "GHA"),
    selected_country_codes: Sequence[str] = ("KEN", "RWA", "GHA", "ZAF", "NGA", "COD"),
    nigeria_peers: Sequence[str] = ("RWA", "UGA", "KEN", "GHA"),
    future_horizon: int = 10,
) -> list[str]:
    """Run a configurable list of plot routines and return output paths."""
    outputs: list[str] = []
    for raw_name in names:
        name = raw_name.strip().lower()
        if not name:
            continue
        try:
            result = None
            if name == "choropleth":
                result = plotter.plot_choropleth()
            elif name == "spaghetti":
                result = plotter.plot_spaghetti()
            elif name == "pillar_heatmap":
                result = plotter.plot_pillar_heatmap()
            elif name == "box":
                result = plotter.plot_box()
            elif name == "gdp_scatter":
                result = plotter.plot_gdp_scatter()
            elif name == "gdp_scatter_2020":
                result = plotter.plot_gdp_scatter(year=2020)
            elif name == "gdp_scatter_highlight":
                result = plotter.plot_gdp_scatter(highlight_codes=("NGA", "ZAF", "RWA"), filename="plot_gdp_scatter_highlight.png")
            elif name == "gdp_scatter_top10":
                result = plotter.plot_gdp_scatter(year=plotter.latest_year, annotate_top_n=10, filename=f"plot_gdp_scatter_top10_{plotter.latest_year}.png")
            elif name == "corr":
                result = plotter.plot_corr()
            elif name in {"corr_lower", "corr_lt"}:
                result = plotter.plot_corr(lower_triangle=True)
            elif name == "slope":
                result = plotter.plot_slope()
            elif name == "radar":
                result = plotter.plot_radar()
            elif name == "umap":
                result = plotter.plot_umap()
            elif name == "forecast_diag":
                for code in forecast_diag_countries:
                    outputs.append(plotter.plot_forecast_diag(code, weights_file, assets_file))
                continue
            elif name == "forecast_diag_zaf":
                result = plotter.plot_forecast_diag("ZAF", weights_file, assets_file)
            elif name == "nigeria_comparison":
                result = plotter.plot_nigeria_comparison(nigeria_peers)
            elif name == "nigeria_comparison_seaborn":
                result = plotter.plot_nigeria_comparison_seaborn(nigeria_peers)
            elif name == "spaghetti_selected":
                result = plotter.plot_spaghetti_selected(selected_country_codes)
            elif name == "spaghetti_selected_alt":
                result = plotter.plot_spaghetti_selected(selected_country_codes, filename="plot_spaghetti_selected_1.png")
            elif name == "future_projection":
                _, result = plotter.plot_future_skills_predictions(future_horizon, weights_file, assets_file)
            else:
                print(f"[WARN] Unknown plot name: {raw_name}")

            if result:
                outputs.append(result)
        except Exception as exc:
            print(f"[WARN] Plot '{raw_name}' skipped: {exc}")
    return outputs
