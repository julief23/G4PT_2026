from unittest.mock import patch

import pandas as pd
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse


class AGAPEWebsiteTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_simulation_page_loads(self):
        response = self.client.get(reverse("simulator:simulation"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "form")

    def test_results_page_loads(self):
        response = self.client.get(reverse("simulator:results"))
        self.assertEqual(response.status_code, 200)

    def test_invalid_smiles_is_rejected(self):
        response = self.client.post(reverse("simulator:run_simulation"), {
            "job_name": "invalid_test",
            "modelType": "ML",
            "classicalInputType": "smiles",
            "smiles": "%%%%",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid SMILES")

    @patch("simulator.views.compute_mordred_from_smiles_list")
    @patch("simulator.views.predictor.predict")
    def test_valid_smiles_prediction_works(self, mock_predict, mock_mordred):
        mock_mordred.return_value = pd.DataFrame({
            "descriptor_1": [1.0],
            "descriptor_2": [2.0],
        })

        mock_predict.return_value = (
            [1],          # prediction: ACTIVE
            [0.92],       # probability active
            0.0,          # imputation percent
            ["0/2"],      # per-row imputation
        )

        response = self.client.post(reverse("simulator:run_simulation"), {
            "job_name": "valid_test",
            "modelType": "ML",
            "classicalInputType": "smiles",
            "smiles": "CCO",
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ACTIVE")
        self.assertContains(response, "CCO")
        self.assertContains(response, "0.9200")

    def test_csv_without_smiles_column_is_rejected(self):
        csv_content = b"name,value\nmol1,3\nmol2,5\n"

        uploaded_file = SimpleUploadedFile(
            "bad_file.csv",
            csv_content,
            content_type="text/csv"
        )

        response = self.client.post(reverse("simulator:run_simulation"), {
            "job_name": "bad_csv_test",
            "modelType": "ML",
            "classicalInputType": "csv",
            "classicalCsv": uploaded_file,
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "CSV must contain")

    @patch("simulator.views.compute_mordred_from_smiles_list")
    @patch("simulator.views.predictor.predict")
    def test_download_prediction_csv_works(self, mock_predict, mock_mordred):
        mock_mordred.return_value = pd.DataFrame({
            "descriptor_1": [1.0],
            "descriptor_2": [2.0],
        })

        mock_predict.return_value = (
            [1],
            [0.91],
            0.0,
            ["0/2"],
        )

        self.client.post(reverse("simulator:run_simulation"), {
            "job_name": "download_test",
            "modelType": "ML",
            "classicalInputType": "smiles",
            "smiles": "CCO",
        })

        response = self.client.get(reverse("simulator:download_prediction"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertIn("download_test_predictions.csv", response["Content-Disposition"])
        self.assertContains(response, "ACTIVE")