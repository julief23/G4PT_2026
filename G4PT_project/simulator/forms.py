from django import forms

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

    draw = forms.CharField(required=False)

    def clean(self):
        cleaned_data = super().clean()

        d = (cleaned_data.get("draw") or "").strip()
        if not d:
            self.add_error("draw", "Please draw a molecule.")

        return cleaned_data