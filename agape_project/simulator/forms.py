from django import forms
from simulator.services.csv_utils import robust_csv_reader

class SimulationForm(forms.Form):

    job_name = forms.CharField(
        max_length=100,
        required=True,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "e.g., Binder-G4 complex",
            "translate": "no"
        })
    )

    MODEL_CHOICES = [
        ("DNN", "Deep neural network Model"),
        ("ML", "XGboost Model"),
    ]

    modelType = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.RadioSelect
    )

    INPUT_CHOICES = [
        ("csv", "CSV Upload"),
        ("smiles", "Single SMILES"),
        ("draw", "Draw Molecule"),
    ]

    classicalInputType = forms.ChoiceField(
        choices=INPUT_CHOICES,
        widget=forms.RadioSelect
    )

    classicalCsv = forms.FileField(required=False)
    smiles = forms.CharField(required=False)
    draw = forms.CharField(required=False)

    def clean(self):
        cleaned_data = super().clean()
        input_mode = cleaned_data.get("classicalInputType")

        # ---------- CSV MODE ----------
        if input_mode == "csv":
            f = cleaned_data.get("classicalCsv")

            if not f:
                self.add_error("classicalCsv", "Please upload a CSV file.")
                return cleaned_data

            if not f.name.lower().endswith(".csv"):
                self.add_error("classicalCsv", "File must be a .csv file.")
                return cleaned_data

            try:
                df = robust_csv_reader(f)
            except Exception:
                self.add_error("classicalCsv", "Invalid or unreadable CSV file.")
                return cleaned_data

            if df.empty:
                self.add_error("classicalCsv", "CSV file is empty.")
                return cleaned_data

            if len(df) > 10000:
                self.add_error("classicalCsv", "Too many rows (max 10,000).")
                return cleaned_data

            cols = [c.lower().strip() for c in df.columns]
            if "smiles" not in cols:
                self.add_error("classicalCsv", "CSV must contain a SMILES column.")
                return cleaned_data

            cleaned_data["csv_df"] = df  

        # ---------- SINGLE SMILES MODE ----------
        elif input_mode == "smiles":
            s = (cleaned_data.get("smiles") or "").strip()
            if not s:
                self.add_error("smiles", "Please enter a SMILES string.")
                return cleaned_data
            cleaned_data["smiles"] = s  

        # ---------- DRAW MODE ----------
        elif input_mode == "draw":
            d = (cleaned_data.get("draw") or "").strip()
            if not d:
                self.add_error("draw", "Please draw a molecule.")
                return cleaned_data
            cleaned_data["draw"] = d  

        return cleaned_data