import io
import os
import unittest
from unittest import mock

import fitz
from openpyxl import load_workbook

from app import DEFAULT_OPENAI_MODEL, ExpenseData, ExpenseReportApp


class ExpenseReportAppTests(unittest.TestCase):
    def setUp(self):
        self.app = ExpenseReportApp.__new__(ExpenseReportApp)

    def test_password_helpers(self):
        self.assertTrue(ExpenseReportApp.password_matches("shared", "shared"))
        self.assertFalse(ExpenseReportApp.password_matches("wrong", "shared"))
        self.assertFalse(ExpenseReportApp.password_matches("", "shared"))

        with mock.patch("app.st.secrets.get", return_value=None):
            with mock.patch.dict(os.environ, {"APP_PASSWORD": " env-secret "}):
                self.assertEqual(
                    ExpenseReportApp.get_configured_app_password(), "env-secret"
                )

        with mock.patch("app.st.secrets.get", return_value="secret-value"):
            with mock.patch.dict(os.environ, {"APP_PASSWORD": "env-secret"}):
                self.assertEqual(
                    ExpenseReportApp.get_configured_app_password(), "secret-value"
                )

        with mock.patch("app.st.secrets.get", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(ExpenseReportApp.get_configured_app_password())

    def test_openai_model_config(self):
        with mock.patch("app.st.secrets.get", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(ExpenseReportApp.get_openai_model(), DEFAULT_OPENAI_MODEL)

        with mock.patch("app.st.secrets.get", return_value=None):
            with mock.patch.dict(os.environ, {"OPENAI_MODEL": " gpt-test-env "}):
                self.assertEqual(ExpenseReportApp.get_openai_model(), "gpt-test-env")

        with mock.patch("app.st.secrets.get", return_value="gpt-test-secret"):
            with mock.patch.dict(os.environ, {"OPENAI_MODEL": "gpt-test-env"}):
                self.assertEqual(
                    ExpenseReportApp.get_openai_model(), "gpt-test-secret"
                )

    def test_email_validation(self):
        self.assertTrue(self.app.is_valid_email("person@berkeley.edu"))
        self.assertFalse(self.app.is_valid_email("guest@example.com"))
        self.assertTrue(
            self.app.is_valid_email("guest@example.com", allow_external=True)
        )

    def test_food_delivery_category_override(self):
        category, expense_type, meal_type = self.app.auto_correct_category(
            "Uber Eats delivery for seminar dinner",
            "Other Business Expenses",
            None,
        )

        self.assertEqual(category, "Meal")
        self.assertEqual(expense_type, "DAILY")
        self.assertEqual(meal_type, "INCIDENTAL")

    def test_membership_category_override(self):
        category, expense_type, meal_type = self.app.auto_correct_category(
            "ASA professional membership dues",
            "Other Business Expenses",
            None,
        )

        self.assertEqual(category, "Conference/Event Registration")
        self.assertEqual(expense_type, "MISCELLANEOUS")
        self.assertIsNone(meal_type)

    def test_duplicate_expense_detection(self):
        expenses = [
            ExpenseData(
                amount=12.50,
                currency="USD",
                description="Coffee",
                date="2026-04-14",
                category="Meal",
                filename="one.pdf",
            ),
            ExpenseData(
                amount=12.50,
                currency="USD",
                description="Coffee duplicate",
                date="2026-04-14",
                category="Meal",
                filename="two.pdf",
            ),
        ]

        duplicates = self.app.detect_duplicate_expenses(expenses)

        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0][0].filename, "one.pdf")
        self.assertEqual(duplicates[0][1].filename, "two.pdf")

    def test_generate_xlsx_summary(self):
        expenses = [
            ExpenseData(
                amount=20.00,
                currency="USD",
                description="Conference lunch",
                date="2026-04-14",
                category="Meal",
                filename="lunch.pdf",
                meal_type="LUNCH",
            )
        ]
        metadata = {
            "first_name": "Ada",
            "last_name": "Lovelace",
            "description": "Attending research seminar",
            "start_date": "04/14/2026",
            "end_date": "04/14/2026",
            "visa_status": "N/A",
            "visa_us_check": "N/A",
            "visa_wire_transfer": "N/A",
            "exchange_rates": {"USD": 1.0},
        }

        xlsx_bytes = self.app.generate_xlsx_summary(expenses, metadata)
        workbook = load_workbook(io.BytesIO(xlsx_bytes), data_only=False)
        sheet = workbook["Expenses"]

        self.assertEqual(sheet["B1"].value, "Ada Lovelace")
        self.assertEqual(sheet["B2"].value, "Attending research seminar")
        self.assertEqual(sheet["D29"].value, "Conference lunch")

    def test_redacts_personal_email_in_pdf_body_text(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Please reimburse Jane. Personal contact: jane.doe@gmail.com",
        )
        page.insert_text((72, 96), "Department contact: travel@berkeley.edu")
        source_pdf = doc.tobytes()
        doc.close()

        redacted_pdf = self.app.redact_pii_from_pdf(source_pdf)
        redacted_doc = fitz.open(stream=redacted_pdf, filetype="pdf")
        redacted_text = "\n".join(page.get_text() for page in redacted_doc)
        page_count = redacted_doc.page_count
        redacted_doc.close()

        self.assertNotIn("jane.doe@gmail.com", redacted_text)
        self.assertEqual(page_count, 1)


if __name__ == "__main__":
    unittest.main()
