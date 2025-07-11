import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import json
from typing import List, Dict, Any

class KathmanduCooperativeLogGenerator:
    def __init__(self):
        # Cooperative-specific data for Kathmandu Valley
        self.cooperatives = [
            "Nepal Cooperative Bank",
            "Kathmandu Valley Cooperative",
            "Baneshwor Cooperative Society",
            "Patan Cooperative Bank",
            "Bhaktapur Cooperative",
            "Lalitpur Cooperative Society",
            "Kirtipur Cooperative Bank",
            "Thimi Cooperative Society"
        ]
        
        self.branches = {
            "Nepal Cooperative Bank": ["New Baneshwor", "Thamel", "Patan", "Bhaktapur", "Kirtipur"],
            "Kathmandu Valley Cooperative": ["Kathmandu Central", "Lalitpur", "Bhaktapur", "Kirtipur"],
            "Baneshwor Cooperative Society": ["Baneshwor", "New Baneshwor", "Old Baneshwor"],
            "Patan Cooperative Bank": ["Patan Durbar", "Jawalakhel", "Kupondole", "Pulchowk"],
            "Bhaktapur Cooperative": ["Bhaktapur Durbar", "Thimi", "Suryabinayak"],
            "Lalitpur Cooperative Society": ["Lalitpur Central", "Patan", "Jawalakhel"],
            "Kirtipur Cooperative Bank": ["Kirtipur Central", "Panga", "Chobhar"],
            "Thimi Cooperative Society": ["Thimi Central", "Suryabinayak", "Naghade"]
        }
        
        self.transaction_types = [
            "deposit", "withdraw", "transfer", "loan_payment", "interest_credit",
            "fee_charge", "account_opening", "account_closing", "balance_inquiry"
        ]
        
        self.locations = [
            "Kathmandu", "Lalitpur", "Bhaktapur", "Kirtipur", "Thimi", "Patan",
            "Baneshwor", "Thamel", "Jawalakhel", "Kupondole", "Pulchowk"
        ]
        
        # IP ranges for Kathmandu Valley
        self.ip_ranges = [
            "192.168.1.", "192.168.2.", "10.0.1.", "10.0.2.", "172.16.1.",
            "172.16.2.", "192.168.10.", "192.168.20.", "10.1.1.", "10.1.2."
        ]
        
        # Member ID prefixes for different cooperatives
        self.member_prefixes = {
            "Nepal Cooperative Bank": "NCB",
            "Kathmandu Valley Cooperative": "KVC", 
            "Baneshwor Cooperative Society": "BCS",
            "Patan Cooperative Bank": "PCB",
            "Bhaktapur Cooperative": "BHC",
            "Lalitpur Cooperative Society": "LCS",
            "Kirtipur Cooperative Bank": "KCB",
            "Thimi Cooperative Society": "TCS"
        }
        
        # Account number patterns
        self.account_patterns = {
            "Nepal Cooperative Bank": "NCB",
            "Kathmandu Valley Cooperative": "KVC",
            "Baneshwor Cooperative Society": "BCS", 
            "Patan Cooperative Bank": "PCB",
            "Bhaktapur Cooperative": "BHC",
            "Lalitpur Cooperative Society": "LCS",
            "Kirtipur Cooperative Bank": "KCB",
            "Thimi Cooperative Society": "TCS"
        }

    def generate_member_id(self, cooperative: str) -> str:
        """Generate realistic member ID based on cooperative"""
        prefix = self.member_prefixes.get(cooperative, "COOP")
        member_num = random.randint(1000, 9999)
        return f"{prefix}{member_num:04d}"

    def generate_account_number(self, cooperative: str) -> str:
        """Generate realistic account number"""
        prefix = self.account_patterns.get(cooperative, "COOP")
        account_num = random.randint(100000, 999999)
        return f"{prefix}{account_num}XXXX"

    def generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate random timestamp within given range"""
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_seconds = random.randrange(24 * 3600)
        return start_date + timedelta(days=random_days, seconds=random_seconds)

    def generate_ip_address(self) -> str:
        """Generate realistic IP address"""
        ip_range = random.choice(self.ip_ranges)
        last_octet = random.randint(1, 254)
        return f"{ip_range}{last_octet}"

    def generate_amount(self, transaction_type: str) -> float:
        """Generate realistic transaction amounts based on type"""
        if transaction_type == "deposit":
            return round(random.uniform(100, 50000), 2)
        elif transaction_type == "withdraw":
            return round(random.uniform(50, 15000), 2)
        elif transaction_type == "transfer":
            return round(random.uniform(100, 25000), 2)
        elif transaction_type == "loan_payment":
            return round(random.uniform(500, 10000), 2)
        elif transaction_type == "interest_credit":
            return round(random.uniform(10, 500), 2)
        elif transaction_type == "fee_charge":
            return round(random.uniform(5, 200), 2)
        else:
            return 0.0

    def generate_balance(self, previous_balance: float, amount: float, transaction_type: str) -> float:
        """Calculate balance after transaction"""
        if transaction_type in ["deposit", "interest_credit"]:
            return round(previous_balance + amount, 2)
        elif transaction_type in ["withdraw", "transfer", "loan_payment", "fee_charge"]:
            return round(previous_balance - amount, 2)
        else:
            return previous_balance

    def generate_anomaly_label(self, transaction_type: str, amount: float, balance: float) -> int:
        """Generate anomaly labels based on suspicious patterns"""
        # Define anomaly conditions
        anomalies = []
        
        # Large withdrawal relative to balance
        if transaction_type == "withdraw" and amount > balance * 0.8:
            anomalies.append(True)
        
        # Very large amounts
        if amount > 100000:
            anomalies.append(True)
        
        # Multiple transactions in short time (simulated)
        if random.random() < 0.05:  # 5% chance of anomaly
            anomalies.append(True)
        
        # Unusual transaction patterns
        if transaction_type == "transfer" and amount > 50000:
            anomalies.append(True)
        
        return 1 if any(anomalies) else 0

    def generate_single_log(self, log_id: int, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate a single log entry"""
        cooperative = random.choice(self.cooperatives)
        member_id = self.generate_member_id(cooperative)
        account_no = self.generate_account_number(cooperative)
        transaction_type = random.choice(self.transaction_types)
        amount = self.generate_amount(transaction_type)
        
        # Generate realistic balance
        previous_balance = random.uniform(1000, 100000)
        balance = self.generate_balance(previous_balance, amount, transaction_type)
        
        # Ensure balance doesn't go negative for normal transactions
        if balance < 0 and transaction_type in ["withdraw", "transfer"]:
            amount = previous_balance * 0.9
            balance = self.generate_balance(previous_balance, amount, transaction_type)
        
        branch_name = random.choice(self.branches[cooperative])
        device_ip = self.generate_ip_address()
        location = random.choice(self.locations)
        
        # Generate anomaly label
        label = self.generate_anomaly_label(transaction_type, amount, balance)
        
        return {
            "log_id": log_id,
            "timestamp": self.generate_timestamp(start_date, end_date),
            "member_id": member_id,
            "account_no": account_no,
            "transaction_type": transaction_type,
            "amount": amount,
            "balance": balance,
            "branch_name": branch_name,
            "device_ip": device_ip,
            "location": location,
            "label": label
        }

    def generate_logs(self, num_logs: int = 1000, days_back: int = 30) -> pd.DataFrame:
        """Generate specified number of logs"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logs = []
        for i in range(num_logs):
            log = self.generate_single_log(i + 1, start_date, end_date)
            logs.append(log)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(logs)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df

    def save_logs(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save logs to CSV and JSON files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kathmandu_cooperative_logs_{timestamp}"
        
        # Save as CSV
        csv_filename = f"{filename}.csv"
        df.to_csv(csv_filename, index=False)
        
        # Save as JSON
        json_filename = f"{filename}.json"
        df.to_json(json_filename, orient='records', indent=2, date_format='iso')
        
        return csv_filename, json_filename

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics of generated logs"""
        print("\n" + "="*60)
        print("KATHMANDU VALLEY COOPERATIVE LOGS SUMMARY")
        print("="*60)
        print(f"Total Logs Generated: {len(df)}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total Anomalies: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
        print(f"Total Transaction Amount: NPR {df['amount'].sum():,.2f}")
        print(f"Average Transaction Amount: NPR {df['amount'].mean():,.2f}")
        
        print("\nTransaction Types Distribution:")
        print(df['transaction_type'].value_counts())
        
        print("\nCooperatives Distribution:")
        cooperative_counts = df['branch_name'].map(lambda x: next((k for k, v in self.branches.items() if x in v), 'Unknown'))
        print(cooperative_counts.value_counts())
        
        print("\nLocations Distribution:")
        print(df['location'].value_counts())

def main():
    """Main function to generate and save cooperative logs"""
    print("Generating Kathmandu Valley Cooperative Transaction Logs...")
    
    # Initialize generator
    generator = KathmanduCooperativeLogGenerator()
    
    # Generate logs (1000 logs for last 30 days)
    df = generator.generate_logs(num_logs=1000, days_back=30)
    
    # Save logs
    csv_file, json_file = generator.save_logs(df)
    
    # Print summary
    generator.print_summary(df)
    
    print(f"\nLogs saved to:")
    print(f"CSV: {csv_file}")
    print(f"JSON: {json_file}")
    
    # Display first few logs
    print("\nFirst 5 Log Entries:")
    print(df.head().to_string(index=False))
    
    # Display anomaly examples
    anomalies = df[df['label'] == 1]
    if not anomalies.empty:
        print(f"\nSample Anomaly Logs ({len(anomalies)} found):")
        print(anomalies.head().to_string(index=False))

if __name__ == "__main__":
    main() 