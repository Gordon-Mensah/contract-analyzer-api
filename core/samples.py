# core/samples.py

def get_sample_contract(contract_type):
    samples = {
        "nda": """
        This Non-Disclosure Agreement is made between Company and Tester. Tester agrees not to disclose any confidential information including trade secrets, source code, or test results. Tester shall not copy, reverse engineer, or share the software with third parties. This agreement terminates upon request or after 30 days.
        """,
        "rental": """
        This Rental Agreement is between Landlord and Tenant. Rent is $1200/month due on the 1st. Tenant shall maintain the property and report damages. Either party may terminate the lease with 30 days' notice. No pets allowed. Security deposit of $1000 required.
        """,
        "employment": """
        This Employment Contract is between Employer and Employee. Employee will serve as Marketing Manager with a salary of $60,000/year. Benefits include health insurance and paid leave. Termination requires 2 weeks' notice. All work products are owned by the company.
        """,
        "service": """
        This Service Agreement is between Client and Provider. Provider will deliver website design services by June 1st. Total fee is $3000 payable in two installments. Provider is not liable for delays caused by third parties. Either party may terminate with 10 days' notice.
        """,
        "sales": """
        This Sales Agreement is between Seller and Buyer. Buyer agrees to purchase 100 units of Product X at $25/unit. Delivery will occur within 14 days. Warranty covers defects for 90 days. Returns accepted within 30 days with receipt.
        """,
        "other": """
        This Agreement outlines general terms between two parties. Each party agrees to act in good faith. Disputes will be resolved under the laws of the State of New York. Amendments must be in writing and signed by both parties.
        """
    }
    return samples.get(contract_type, samples["other"]).strip()
