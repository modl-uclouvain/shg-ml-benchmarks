from re2fractive import OptimadeDataset


class GNome2025Dataset(OptimadeDataset):
    properties = {"band_gap": "_gnome_bandgap", "hull_distance": "_gnome_hull_distance"}
    id = "GNome2025"
    filter = "_gnome_band_gap > 0.05 AND _gnome_hull_distance < 0.1"


class Alexandria2025(OptimadeDataset):
    properties = {
        "band_gap": "_alexandria_band_gap",
        "hull_distance": "_alexandria_hull_distance",
    }
    id: str = "Alexandria2024"
    base_url: str = "https://alexandria.icams.rub.de/pbe"
    filter: str = "_alexandria_band_gap > 0.05 AND _alexandria_hull_distance <= 0.025"
