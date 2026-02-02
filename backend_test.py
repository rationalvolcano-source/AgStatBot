#!/usr/bin/env python3
"""
AGstat Lite Backend API Testing Suite
Tests all endpoints for the agricultural biostatistics app
"""

import requests
import json
import sys
import io
import pandas as pd
from datetime import datetime
from pathlib import Path

class AGstatAPITester:
    def __init__(self, base_url="https://cropstat-assist.preview.emergentagent.com"):
        self.base_url = base_url
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tests_run = 0
        self.tests_passed = 0
        self.data_id = None
        self.failed_tests = []

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
            self.failed_tests.append({"test": name, "error": details})

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            details = f"Expected {expected_status}, got {response.status_code}"
            
            if not success and response.text:
                try:
                    error_data = response.json()
                    details += f" - {error_data.get('detail', response.text[:100])}"
                except:
                    details += f" - {response.text[:100]}"

            self.log_test(name, success, details if not success else "")
            
            return success, response.json() if success and response.text else {}

        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def create_test_csv(self):
        """Create a test CSV file for upload testing"""
        # Create sample agricultural data
        data = {
            'Treatment': ['Control', 'Fertilizer_A', 'Fertilizer_B', 'Control', 'Fertilizer_A', 'Fertilizer_B'] * 5,
            'Block': ['Block1', 'Block1', 'Block1', 'Block2', 'Block2', 'Block2'] * 5,
            'Yield': [45.2, 52.1, 48.7, 43.8, 50.9, 47.3, 44.1, 53.2, 49.1, 42.9, 51.8, 48.5,
                     46.0, 52.8, 47.9, 44.5, 51.2, 48.1, 45.3, 52.5, 48.8, 43.2, 50.7, 47.6,
                     44.8, 53.0, 49.2, 43.5, 51.5, 48.0],
            'Height': [120.5, 135.2, 128.1, 118.9, 133.7, 126.8, 121.3, 136.1, 129.4, 119.2, 134.5, 127.2,
                      122.1, 135.8, 128.7, 120.0, 134.2, 127.9, 121.7, 135.5, 128.3, 119.8, 133.9, 127.5,
                      120.8, 136.0, 129.1, 119.5, 134.8, 128.0],
            'Protein': [12.1, 14.5, 13.2, 11.8, 14.2, 12.9, 12.3, 14.7, 13.5, 11.9, 14.1, 13.0,
                       12.0, 14.6, 13.3, 12.2, 14.3, 13.1, 12.4, 14.4, 13.4, 11.7, 14.0, 12.8,
                       12.5, 14.8, 13.6, 12.1, 14.2, 13.2]
        }
        
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        return success and response.get('status') == 'healthy'

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET", 
            "api/",
            200
        )
        return success

    def test_file_upload(self):
        """Test file upload functionality"""
        csv_data = self.create_test_csv()
        files = {'file': ('test_data.csv', csv_data, 'text/csv')}
        
        success, response = self.run_test(
            "File Upload",
            "POST",
            "api/upload",
            200,
            files=files
        )
        
        if success and 'id' in response:
            self.data_id = response['id']
            print(f"   📁 Uploaded data ID: {self.data_id}")
            print(f"   📊 Data: {response.get('rows', 0)} rows, {len(response.get('columns', []))} columns")
            return True
        return False

    def test_data_retrieval(self):
        """Test data retrieval endpoint"""
        if not self.data_id:
            self.log_test("Data Retrieval", False, "No data ID available")
            return False
            
        success, response = self.run_test(
            "Data Retrieval",
            "GET",
            f"api/data/{self.data_id}",
            200
        )
        return success

    def test_chat_endpoint(self):
        """Test chat functionality"""
        chat_data = {
            "session_id": self.session_id,
            "message": "Hello, can you help me analyze my agricultural data?",
            "data_id": self.data_id
        }
        
        success, response = self.run_test(
            "Chat Endpoint",
            "POST",
            "api/chat",
            200,
            data=chat_data
        )
        
        if success:
            print(f"   💬 Chat response: {response.get('content', '')[:100]}...")
        return success

    def test_anova_analysis(self):
        """Test ANOVA analysis endpoint"""
        if not self.data_id:
            self.log_test("ANOVA Analysis", False, "No data ID available")
            return False
            
        anova_params = {
            "data_id": self.data_id,
            "analysis_type": "anova",
            "parameters": {
                "dependent_var": "Yield",
                "independent_var": "Treatment",
                "block_var": "Block"
            }
        }
        
        success, response = self.run_test(
            "ANOVA Analysis",
            "POST",
            "api/analyze/anova",
            200,
            data=anova_params
        )
        
        if success:
            print(f"   📈 ANOVA completed with plot: {'Yes' if response.get('plot_base64') else 'No'}")
        return success

    def test_pca_analysis(self):
        """Test PCA analysis endpoint"""
        if not self.data_id:
            self.log_test("PCA Analysis", False, "No data ID available")
            return False
            
        pca_params = {
            "data_id": self.data_id,
            "analysis_type": "pca",
            "parameters": {
                "numeric_cols": ["Yield", "Height", "Protein"],
                "group_col": "Treatment",
                "n_components": 2
            }
        }
        
        success, response = self.run_test(
            "PCA Analysis",
            "POST",
            "api/analyze/pca",
            200,
            data=pca_params
        )
        
        if success:
            results = response.get('results', {})
            variance_explained = sum(results.get('explained_variance_ratio', []))
            print(f"   🔍 PCA variance explained: {variance_explained*100:.1f}%")
        return success

    def test_clustering_analysis(self):
        """Test clustering analysis endpoint"""
        if not self.data_id:
            self.log_test("Clustering Analysis", False, "No data ID available")
            return False
            
        clustering_params = {
            "data_id": self.data_id,
            "analysis_type": "clustering",
            "parameters": {
                "numeric_cols": ["Yield", "Height", "Protein"],
                "n_clusters": 3,
                "method": "kmeans"
            }
        }
        
        success, response = self.run_test(
            "Clustering Analysis",
            "POST",
            "api/analyze/clustering",
            200,
            data=clustering_params
        )
        
        if success:
            results = response.get('results', {})
            silhouette = results.get('silhouette_score', 0)
            print(f"   🎯 Clustering silhouette score: {silhouette:.3f}")
        return success

    def test_descriptive_analysis(self):
        """Test descriptive statistics endpoint"""
        if not self.data_id:
            self.log_test("Descriptive Analysis", False, "No data ID available")
            return False
            
        desc_params = {
            "data_id": self.data_id,
            "analysis_type": "descriptive",
            "parameters": {
                "columns": ["Yield", "Height", "Protein"],
                "group_by": "Treatment"
            }
        }
        
        success, response = self.run_test(
            "Descriptive Analysis",
            "POST",
            "api/analyze/descriptive",
            200,
            data=desc_params
        )
        return success

    def test_correlation_analysis(self):
        """Test correlation analysis endpoint"""
        if not self.data_id:
            self.log_test("Correlation Analysis", False, "No data ID available")
            return False
            
        corr_params = {
            "data_id": self.data_id,
            "analysis_type": "correlation",
            "parameters": {
                "columns": ["Yield", "Height", "Protein"],
                "method": "pearson"
            }
        }
        
        success, response = self.run_test(
            "Correlation Analysis",
            "POST",
            "api/analyze/correlation",
            200,
            data=corr_params
        )
        return success

    def test_normality_analysis(self):
        """Test normality testing endpoint"""
        if not self.data_id:
            self.log_test("Normality Analysis", False, "No data ID available")
            return False
            
        norm_params = {
            "data_id": self.data_id,
            "analysis_type": "normality",
            "parameters": {
                "column": "Yield",
                "group_by": "Treatment"
            }
        }
        
        success, response = self.run_test(
            "Normality Analysis",
            "POST",
            "api/analyze/normality",
            200,
            data=norm_params
        )
        return success

    def test_code_execution(self):
        """Test code execution endpoint"""
        if not self.data_id:
            self.log_test("Code Execution", False, "No data ID available")
            return False
            
        code_params = {
            "code": "print('Hello from AGstat!')\nprint(f'Data shape: {df.shape}')\nprint(df.head())",
            "data_id": self.data_id
        }
        
        success, response = self.run_test(
            "Code Execution",
            "POST",
            "api/execute",
            200,
            data=code_params
        )
        
        if success:
            print(f"   💻 Code executed successfully: {response.get('success', False)}")
        return success

    def test_chat_history(self):
        """Test chat history endpoints"""
        # Get chat history
        success1, _ = self.run_test(
            "Get Chat History",
            "GET",
            f"api/chat/history/{self.session_id}",
            200
        )
        
        # Clear chat history
        success2, _ = self.run_test(
            "Clear Chat History",
            "DELETE",
            f"api/chat/history/{self.session_id}",
            200
        )
        
        return success1 and success2

    def run_all_tests(self):
        """Run all API tests"""
        print("🧪 Starting AGstat Lite API Tests")
        print("=" * 50)
        
        # Basic endpoint tests
        print("\n📡 Basic Endpoints:")
        self.test_health_endpoint()
        self.test_root_endpoint()
        
        # File upload and data tests
        print("\n📁 Data Management:")
        upload_success = self.test_file_upload()
        if upload_success:
            self.test_data_retrieval()
        
        # Chat functionality
        print("\n💬 Chat Functionality:")
        self.test_chat_endpoint()
        self.test_chat_history()
        
        # Analysis endpoints (only if data upload succeeded)
        if upload_success:
            print("\n📊 Statistical Analysis:")
            self.test_anova_analysis()
            self.test_pca_analysis()
            self.test_clustering_analysis()
            self.test_descriptive_analysis()
            self.test_correlation_analysis()
            self.test_normality_analysis()
            
            print("\n💻 Code Execution:")
            self.test_code_execution()
        else:
            print("\n⚠️  Skipping analysis tests - file upload failed")
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Summary:")
        print(f"   Total tests: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {len(self.failed_tests)}")
        print(f"   Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test['test']}: {test['error']}")
        
        return len(self.failed_tests) == 0

def main():
    """Main test runner"""
    tester = AGstatAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())