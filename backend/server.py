from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
import io
import base64
import asyncio

# Statistical imports
from scipy import stats
from scipy.stats import shapiro, levene, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# LLM Integration
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Get LLM key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Neon plot style configuration
def set_neon_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#020408',
        'axes.facecolor': '#0B1121',
        'axes.edgecolor': '#00F0FF',
        'axes.labelcolor': '#00F0FF',
        'text.color': '#F8FAFC',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'grid.color': '#1E293B',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'legend.facecolor': '#0B1121',
        'legend.edgecolor': '#00F0FF',
        'font.family': 'sans-serif',
        'font.size': 12
    })

NEON_COLORS = ['#00F0FF', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444', '#22D3EE']

# ==================== Models ====================

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    code: Optional[str] = None
    plot_url: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    data_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    code: Optional[str] = None
    plot_url: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class DataUpload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    columns: List[str]
    rows: int
    preview: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisRequest(BaseModel):
    data_id: str
    analysis_type: str  # 'anova', 'pca', 'clustering', 'normality', 'correlation', 'descriptive'
    parameters: Dict[str, Any]

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_id: str
    analysis_type: str
    results: Dict[str, Any]
    code: str
    plot_base64: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CodeExecutionRequest(BaseModel):
    code: str
    data_id: Optional[str] = None

# ==================== Storage ====================

# In-memory data storage (for uploaded files)
uploaded_data: Dict[str, pd.DataFrame] = {}

# ==================== Helper Functions ====================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#020408')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_anova_code(dependent_var: str, independent_var: str, block_var: Optional[str] = None):
    """Generate Python code for ANOVA analysis"""
    if block_var:
        code = f'''# RCBD ANOVA Analysis
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Fit the model (RCBD)
model = ols('{dependent_var} ~ C({independent_var}) + C({block_var})', data=df).fit()

# ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# Assumption checks
print("\\nShapiro-Wilk Test for Normality (on residuals):")
stat, p = shapiro(model.resid)
print(f"W-statistic: {{stat:.4f}}, p-value: {{p:.4f}}")

# Tukey HSD post-hoc test
tukey = pairwise_tukeyhsd(df['{dependent_var}'], df['{independent_var}'], alpha=0.05)
print("\\nTukey HSD Results:")
print(tukey)'''
    else:
        code = f'''# One-way ANOVA Analysis
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Fit the model
model = ols('{dependent_var} ~ C({independent_var})', data=df).fit()

# ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# Assumption checks
print("\\nShapiro-Wilk Test for Normality (on residuals):")
stat, p = shapiro(model.resid)
print(f"W-statistic: {{stat:.4f}}, p-value: {{p:.4f}}")

# Levene's Test for Homogeneity of Variances
groups = [group['{dependent_var}'].values for name, group in df.groupby('{independent_var}')]
stat, p = levene(*groups)
print(f"\\nLevene's Test: statistic={{stat:.4f}}, p-value={{p:.4f}}")

# Tukey HSD post-hoc test
tukey = pairwise_tukeyhsd(df['{dependent_var}'], df['{independent_var}'], alpha=0.05)
print("\\nTukey HSD Results:")
print(tukey)'''
    return code

def generate_pca_code(numeric_cols: List[str], group_col: Optional[str] = None):
    """Generate Python code for PCA analysis"""
    cols_str = str(numeric_cols)
    code = f'''# PCA Analysis
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select numeric columns
numeric_cols = {cols_str}
X = df[numeric_cols].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=min(len(numeric_cols), len(X)))
components = pca.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{{i+1}}: {{var*100:.2f}}%")
print(f"\\nCumulative: {{sum(pca.explained_variance_ratio_)*100:.2f}}%")

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{{i+1}}' for i in range(len(pca.components_))],
    index=numeric_cols
)
print("\\nPCA Loadings:")
print(loadings)'''
    return code

def generate_clustering_code(numeric_cols: List[str], n_clusters: int = 3, method: str = 'kmeans'):
    """Generate Python code for clustering analysis"""
    cols_str = str(numeric_cols)
    if method == 'kmeans':
        code = f'''# K-Means Clustering Analysis
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Select and scale data
numeric_cols = {cols_str}
X = df[numeric_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters={n_clusters}, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Silhouette score
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {{silhouette:.4f}}")

# Cluster centers
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=numeric_cols
)
print("\\nCluster Centers:")
print(centers)

# Cluster sizes
print("\\nCluster Sizes:")
for i in range({n_clusters}):
    print(f"Cluster {{i}}: {{sum(clusters == i)}} samples")'''
    else:
        code = f'''# Hierarchical Clustering Analysis
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Select and scale data
numeric_cols = {cols_str}
X = df[numeric_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters={n_clusters}, linkage='ward')
clusters = hc.fit_predict(X_scaled)

# Linkage matrix for dendrogram
Z = linkage(X_scaled, method='ward')

print("Cluster Assignments:")
for i in range({n_clusters}):
    print(f"Cluster {{i}}: {{sum(clusters == i)}} samples")'''
    return code

# ==================== API Routes ====================

@api_router.get("/")
async def root():
    return {"message": "AGstat Lite API - Agricultural Biostatistics Assistant"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AGstat Lite"}

# Chat endpoints
@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the AI assistant"""
    try:
        # Create system message for AGstat personality
        system_message = """You are AGstat Lite — a PhD-level agricultural biostatistics assistant. 
        
Your expertise includes:
- Field trials (RCBD, split-plot, lattice)
- Greenhouse & pot experiments (ANOVA, factorials)
- Multi-location & GxE trials (AMMI, GGE)
- Pesticide dose-response (log-logistic, probit)
- Temporal yield/time-series analyses
- Trait clustering (PCA, HCA, K-means)

When analyzing data:
1. ALWAYS check assumptions first (normality with Shapiro-Wilk, homogeneity with Levene's)
2. Suggest the appropriate statistical test based on the experimental design
3. Provide Python code that can be executed
4. Explain results concisely

For first-time users, be welcoming but slightly sarcastic: "Congratulations on surviving field trials and Excel disasters..."

After that, be efficient and precise. Output statistics and recommendations first, explanations only when asked.

If the user uploads data, help them choose the right analysis based on:
- Number of treatments/factors
- Presence of blocking
- Type of response variable
- Sample size considerations"""

        # Initialize LLM chat
        chat_instance = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=request.session_id,
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        # Get chat history from database
        history = await db.chat_messages.find(
            {"session_id": request.session_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(50)
        
        # Add context about uploaded data if available
        context_message = request.message
        if request.data_id and request.data_id in uploaded_data:
            df = uploaded_data[request.data_id]
            data_info = f"\n\n[Data Context: {df.shape[0]} rows, {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}]"
            context_message += data_info
        
        # Create user message
        user_msg = UserMessage(text=context_message)
        
        # Send message and get response
        response_text = await chat_instance.send_message(user_msg)
        
        # Save user message
        user_doc = {
            "id": str(uuid.uuid4()),
            "session_id": request.session_id,
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.chat_messages.insert_one(user_doc)
        
        # Save assistant response
        assistant_id = str(uuid.uuid4())
        assistant_doc = {
            "id": assistant_id,
            "session_id": request.session_id,
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.chat_messages.insert_one(assistant_doc)
        
        return ChatResponse(
            id=assistant_id,
            role="assistant",
            content=response_text
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    messages = await db.chat_messages.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(100)
    return {"messages": messages}

@api_router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    await db.chat_messages.delete_many({"session_id": session_id})
    return {"message": "Chat history cleared"}

# File upload endpoints
@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file for analysis"""
    try:
        content = await file.read()
        filename = file.filename.lower()
        
        # Parse file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Generate unique ID
        data_id = str(uuid.uuid4())
        
        # Store in memory
        uploaded_data[data_id] = df
        
        # Create preview (first 10 rows)
        preview = df.head(10).replace({np.nan: None}).to_dict(orient='records')
        
        # Get column info
        column_info = []
        for col in df.columns:
            col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            unique_count = df[col].nunique()
            column_info.append({
                "name": col,
                "type": col_type,
                "unique_values": unique_count,
                "missing": int(df[col].isna().sum())
            })
        
        # Save metadata to database
        doc = {
            "id": data_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "column_info": column_info,
            "rows": len(df),
            "preview": preview,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.uploads.insert_one(doc)
        
        return {
            "id": data_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "column_info": column_info,
            "rows": len(df),
            "preview": preview
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/{data_id}")
async def get_data_info(data_id: str):
    """Get information about uploaded data"""
    if data_id not in uploaded_data:
        # Try to get from database
        doc = await db.uploads.find_one({"id": data_id}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Data not found")
        return doc
    
    df = uploaded_data[data_id]
    return {
        "id": data_id,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient='records')
    }

# Analysis endpoints
@api_router.post("/analyze/anova")
async def analyze_anova(request: AnalysisRequest):
    """Perform ANOVA analysis"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found. Please upload data first.")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        dependent_var = params.get('dependent_var')
        independent_var = params.get('independent_var')
        block_var = params.get('block_var')
        
        if not dependent_var or not independent_var:
            raise HTTPException(status_code=400, detail="Missing required parameters: dependent_var and independent_var")
        
        # Set neon style
        set_neon_style()
        
        # Fit model
        if block_var:
            formula = f'{dependent_var} ~ C({independent_var}) + C({block_var})'
        else:
            formula = f'{dependent_var} ~ C({independent_var})'
        
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Assumption checks
        shapiro_stat, shapiro_p = shapiro(model.resid)
        
        # Levene's test
        groups = [group[dependent_var].dropna().values for name, group in df.groupby(independent_var)]
        levene_stat, levene_p = levene(*groups)
        
        # Tukey HSD
        tukey = pairwise_tukeyhsd(df[dependent_var].dropna(), 
                                  df.loc[df[dependent_var].notna(), independent_var], 
                                  alpha=0.05)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Boxplot with Tukey letters
        ax1 = axes[0]
        groups_data = df.groupby(independent_var)[dependent_var].apply(list).to_dict()
        positions = range(len(groups_data))
        bp = ax1.boxplot(groups_data.values(), positions=positions, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], NEON_COLORS[:len(groups_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xticklabels(groups_data.keys(), rotation=45, ha='right')
        ax1.set_xlabel(independent_var, fontsize=14, color='#00F0FF')
        ax1.set_ylabel(dependent_var, fontsize=14, color='#00F0FF')
        ax1.set_title('Treatment Comparison (Boxplot)', fontsize=16, color='#00F0FF')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        ax2 = axes[1]
        ax2.scatter(model.fittedvalues, model.resid, c='#00F0FF', alpha=0.6, s=50)
        ax2.axhline(y=0, color='#EF4444', linestyle='--', linewidth=2)
        ax2.set_xlabel('Fitted Values', fontsize=14, color='#00F0FF')
        ax2.set_ylabel('Residuals', fontsize=14, color='#00F0FF')
        ax2.set_title('Residuals vs Fitted', fontsize=16, color='#00F0FF')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        # Generate code
        code = generate_anova_code(dependent_var, independent_var, block_var)
        
        # Prepare results
        anova_results = anova_table.reset_index().to_dict(orient='records')
        tukey_summary = str(tukey)
        
        results = {
            "anova_table": anova_results,
            "assumptions": {
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "normal": bool(shapiro_p > 0.05)
                },
                "levene": {
                    "statistic": float(levene_stat),
                    "p_value": float(levene_p),
                    "homogeneous": bool(levene_p > 0.05)
                }
            },
            "tukey_hsd": tukey_summary,
            "interpretation": f"ANOVA {'is' if shapiro_p > 0.05 and levene_p > 0.05 else 'may not be'} appropriate based on assumption checks."
        }
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "anova",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"ANOVA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze/pca")
async def analyze_pca(request: AnalysisRequest):
    """Perform PCA analysis"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        numeric_cols = params.get('numeric_cols', [])
        group_col = params.get('group_col')
        n_components = params.get('n_components', 2)
        
        if not numeric_cols:
            # Auto-detect numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for PCA")
        
        # Prepare data
        X = df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        n_comp = min(n_components, len(numeric_cols), len(X))
        pca = PCA(n_components=n_comp)
        components = pca.fit_transform(X_scaled)
        
        # Set neon style
        set_neon_style()
        
        # Create biplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scores plot
        ax1 = axes[0]
        if group_col and group_col in df.columns:
            groups = df.loc[X.index, group_col]
            unique_groups = groups.unique()
            for i, grp in enumerate(unique_groups):
                mask = groups == grp
                ax1.scatter(components[mask, 0], components[mask, 1], 
                           c=NEON_COLORS[i % len(NEON_COLORS)], 
                           label=str(grp), s=60, alpha=0.7)
            ax1.legend(title=group_col, loc='upper right')
        else:
            ax1.scatter(components[:, 0], components[:, 1], c='#00F0FF', s=60, alpha=0.7)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, color='#00F0FF')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, color='#00F0FF')
        ax1.set_title('PCA Score Plot', fontsize=16, color='#00F0FF')
        ax1.axhline(y=0, color='#94A3B8', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='#94A3B8', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Loadings plot
        ax2 = axes[1]
        loadings = pca.components_.T
        for i, col in enumerate(numeric_cols):
            ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     head_width=0.05, head_length=0.03, fc='#00F0FF', ec='#00F0FF')
            ax2.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, col, 
                    fontsize=10, color='#F8FAFC', ha='center')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel('PC1 Loading', fontsize=14, color='#00F0FF')
        ax2.set_ylabel('PC2 Loading', fontsize=14, color='#00F0FF')
        ax2.set_title('PCA Loadings (Biplot)', fontsize=16, color='#00F0FF')
        ax2.axhline(y=0, color='#94A3B8', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='#94A3B8', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add circle
        circle = plt.Circle((0, 0), 1, fill=False, color='#00F0FF', linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        # Generate code
        code = generate_pca_code(numeric_cols, group_col)
        
        # Prepare results
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_comp)],
            index=numeric_cols
        )
        
        results = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(sum(pca.explained_variance_ratio_)),
            "loadings": loadings_df.to_dict(),
            "n_components": n_comp
        }
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "pca",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"PCA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze/clustering")
async def analyze_clustering(request: AnalysisRequest):
    """Perform clustering analysis"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        numeric_cols = params.get('numeric_cols', [])
        n_clusters = params.get('n_clusters', 3)
        method = params.get('method', 'kmeans')  # 'kmeans' or 'hierarchical'
        
        if not numeric_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for clustering")
        
        # Prepare data
        X = df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Set neon style
        set_neon_style()
        
        if method == 'kmeans':
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_scaled, clusters)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            ax1 = axes[0]
            for i in range(n_clusters):
                mask = clusters == i
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=NEON_COLORS[i % len(NEON_COLORS)], 
                           label=f'Cluster {i}', s=60, alpha=0.7)
            ax1.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                       pca.transform(kmeans.cluster_centers_)[:, 1],
                       c='white', marker='X', s=200, edgecolors='black', linewidth=2)
            ax1.set_xlabel('PC1', fontsize=14, color='#00F0FF')
            ax1.set_ylabel('PC2', fontsize=14, color='#00F0FF')
            ax1.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=16, color='#00F0FF')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Elbow plot
            ax2 = axes[1]
            inertias = []
            K = range(1, min(10, len(X)))
            for k in K:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            
            ax2.plot(K, inertias, 'o-', color='#00F0FF', linewidth=2, markersize=8)
            ax2.axvline(x=n_clusters, color='#EF4444', linestyle='--', linewidth=2, label=f'Selected k={n_clusters}')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=14, color='#00F0FF')
            ax2.set_ylabel('Inertia', fontsize=14, color='#00F0FF')
            ax2.set_title('Elbow Method', fontsize=16, color='#00F0FF')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            results = {
                "method": "kmeans",
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette),
                "cluster_sizes": {f"Cluster {i}": int(sum(clusters == i)) for i in range(n_clusters)},
                "cluster_centers": scaler.inverse_transform(kmeans.cluster_centers_).tolist()
            }
            
        else:
            # Hierarchical clustering
            from scipy.cluster.hierarchy import dendrogram, linkage
            
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            clusters = hc.fit_predict(X_scaled)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Dendrogram
            ax1 = axes[0]
            Z = linkage(X_scaled, method='ward')
            dendrogram(Z, ax=ax1, color_threshold=0, above_threshold_color='#00F0FF')
            ax1.set_xlabel('Sample Index', fontsize=14, color='#00F0FF')
            ax1.set_ylabel('Distance', fontsize=14, color='#00F0FF')
            ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=16, color='#00F0FF')
            
            # PCA visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            ax2 = axes[1]
            for i in range(n_clusters):
                mask = clusters == i
                ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=NEON_COLORS[i % len(NEON_COLORS)], 
                           label=f'Cluster {i}', s=60, alpha=0.7)
            ax2.set_xlabel('PC1', fontsize=14, color='#00F0FF')
            ax2.set_ylabel('PC2', fontsize=14, color='#00F0FF')
            ax2.set_title(f'Hierarchical Clustering (k={n_clusters})', fontsize=16, color='#00F0FF')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            results = {
                "method": "hierarchical",
                "n_clusters": n_clusters,
                "cluster_sizes": {f"Cluster {i}": int(sum(clusters == i)) for i in range(n_clusters)}
            }
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        # Generate code
        code = generate_clustering_code(numeric_cols, n_clusters, method)
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "clustering",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze/descriptive")
async def analyze_descriptive(request: AnalysisRequest):
    """Generate descriptive statistics"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        columns = params.get('columns', [])
        group_by = params.get('group_by')
        
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Set neon style
        set_neon_style()
        
        if group_by and group_by in df.columns:
            # Grouped statistics
            stats_df = df.groupby(group_by)[columns].agg(['mean', 'std', 'min', 'max', 'count'])
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar plot with error bars for first numeric column
            groups = df[group_by].unique()
            x = np.arange(len(groups))
            means = df.groupby(group_by)[columns[0]].mean()
            stds = df.groupby(group_by)[columns[0]].std()
            
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=NEON_COLORS[:len(groups)], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.set_xlabel(group_by, fontsize=14, color='#00F0FF')
            ax.set_ylabel(columns[0], fontsize=14, color='#00F0FF')
            ax.set_title(f'Mean {columns[0]} by {group_by} (±SD)', fontsize=16, color='#00F0FF')
            ax.grid(True, alpha=0.3, axis='y')
            
            results = {
                "grouped_by": group_by,
                "statistics": stats_df.to_dict()
            }
        else:
            # Overall statistics
            stats_df = df[columns].describe()
            
            # Visualization - histogram for each column
            n_cols = min(len(columns), 4)
            fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            
            for i, col in enumerate(columns[:n_cols]):
                ax = axes[i]
                ax.hist(df[col].dropna(), bins=20, color=NEON_COLORS[i % len(NEON_COLORS)], 
                       alpha=0.7, edgecolor='white')
                ax.axvline(df[col].mean(), color='#EF4444', linestyle='--', linewidth=2, label='Mean')
                ax.set_xlabel(col, fontsize=12, color='#00F0FF')
                ax.set_ylabel('Frequency', fontsize=12, color='#00F0FF')
                ax.set_title(f'Distribution of {col}', fontsize=14, color='#00F0FF')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            results = {
                "statistics": stats_df.to_dict()
            }
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        code = f'''# Descriptive Statistics
import pandas as pd

# Summary statistics
{"grouped_stats = df.groupby('" + group_by + "')[" + str(columns) + "].describe()" if group_by else "stats = df[" + str(columns) + "].describe()"}
print({"grouped_stats" if group_by else "stats"})'''
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "descriptive",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Descriptive stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze/correlation")
async def analyze_correlation(request: AnalysisRequest):
    """Compute correlation matrix"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        columns = params.get('columns', [])
        method = params.get('method', 'pearson')  # 'pearson', 'spearman', 'kendall'
        
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation
        corr_matrix = df[columns].corr(method=method)
        
        # Set neon style
        set_neon_style()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Custom colormap (neon blue to red)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, annot_kws={"size": 10, "color": "white"})
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=16, color='#00F0FF')
        plt.xticks(rotation=45, ha='right', color='#94A3B8')
        plt.yticks(rotation=0, color='#94A3B8')
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        code = f'''# Correlation Analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
columns = {columns}
corr_matrix = df[columns].corr(method='{method}')
print("Correlation Matrix:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('{method.capitalize()} Correlation Matrix')
plt.tight_layout()
plt.show()'''
        
        results = {
            "method": method,
            "correlation_matrix": corr_matrix.to_dict()
        }
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "correlation",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Correlation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze/normality")
async def analyze_normality(request: AnalysisRequest):
    """Test normality of data"""
    try:
        if request.data_id not in uploaded_data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = uploaded_data[request.data_id]
        params = request.parameters
        
        column = params.get('column')
        group_by = params.get('group_by')
        
        if not column:
            raise HTTPException(status_code=400, detail="Column parameter required")
        
        # Set neon style
        set_neon_style()
        
        results = {"tests": {}}
        
        if group_by and group_by in df.columns:
            # Test normality by group
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            groups = df[group_by].unique()
            for grp in groups:
                data = df[df[group_by] == grp][column].dropna()
                if len(data) >= 3:
                    stat, p = shapiro(data)
                    results["tests"][str(grp)] = {
                        "shapiro_statistic": float(stat),
                        "p_value": float(p),
                        "normal": bool(p > 0.05),
                        "n": len(data)
                    }
            
            # Q-Q plot
            ax1 = axes[0]
            for i, grp in enumerate(groups[:5]):
                data = df[df[group_by] == grp][column].dropna()
                stats.probplot(data, dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot', fontsize=16, color='#00F0FF')
            ax1.get_lines()[0].set_color('#00F0FF')
            
            # Histogram by group
            ax2 = axes[1]
            for i, grp in enumerate(groups[:5]):
                data = df[df[group_by] == grp][column].dropna()
                ax2.hist(data, bins=15, alpha=0.5, label=str(grp), 
                        color=NEON_COLORS[i % len(NEON_COLORS)])
            ax2.set_xlabel(column, fontsize=14, color='#00F0FF')
            ax2.set_ylabel('Frequency', fontsize=14, color='#00F0FF')
            ax2.set_title(f'Distribution by {group_by}', fontsize=16, color='#00F0FF')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:
            # Test overall normality
            data = df[column].dropna()
            stat, p = shapiro(data)
            results["tests"]["overall"] = {
                "shapiro_statistic": float(stat),
                "p_value": float(p),
                "normal": p > 0.05,
                "n": len(data)
            }
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Q-Q plot
            ax1 = axes[0]
            stats.probplot(data, dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot', fontsize=16, color='#00F0FF')
            ax1.get_lines()[0].set_color('#00F0FF')
            ax1.get_lines()[1].set_color('#EF4444')
            
            # Histogram with normal curve
            ax2 = axes[1]
            ax2.hist(data, bins=20, density=True, alpha=0.7, color='#00F0FF', edgecolor='white')
            
            # Overlay normal distribution
            mu, std = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, std), color='#EF4444', linewidth=2, label='Normal')
            
            ax2.set_xlabel(column, fontsize=14, color='#00F0FF')
            ax2.set_ylabel('Density', fontsize=14, color='#00F0FF')
            ax2.set_title(f'Distribution of {column}', fontsize=16, color='#00F0FF')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_base64 = fig_to_base64(fig)
        
        code = f'''# Normality Test
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from scipy import stats

# Shapiro-Wilk test
data = df['{column}'].dropna()
stat, p = shapiro(data)
print(f"Shapiro-Wilk Test:")
print(f"W-statistic: {{stat:.4f}}")
print(f"p-value: {{p:.4f}}")
print(f"Normal: {{'Yes' if p > 0.05 else 'No'}} (α=0.05)")

# Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(data, dist="norm", plot=ax1)
ax1.set_title('Q-Q Plot')

# Histogram
ax2.hist(data, bins=20, density=True, alpha=0.7)
ax2.set_title('Histogram')
plt.show()'''
        
        return {
            "id": str(uuid.uuid4()),
            "analysis_type": "normality",
            "results": results,
            "code": code,
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Normality test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Code execution endpoint
@api_router.post("/execute")
async def execute_code(request: CodeExecutionRequest):
    """Execute Python code for custom analysis"""
    try:
        # Prepare execution environment
        local_vars = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'sm': sm
        }
        
        # Add data if available
        if request.data_id and request.data_id in uploaded_data:
            local_vars['df'] = uploaded_data[request.data_id].copy()
        
        # Capture output
        output_buffer = io.StringIO()
        
        # Set neon style for any plots
        set_neon_style()
        
        # Create figure for potential plots
        plt.figure(figsize=(10, 6))
        
        # Execute code
        exec(request.code, {"__builtins__": __builtins__}, local_vars)
        
        # Check if a plot was created
        plot_base64 = None
        if plt.get_fignums():
            fig = plt.gcf()
            plot_base64 = fig_to_base64(fig)
        
        return {
            "success": True,
            "output": output_buffer.getvalue(),
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Download endpoints
@api_router.get("/download/code/{analysis_id}")
async def download_code(analysis_id: str):
    """Download generated code as Python file"""
    # For now, return a sample code file
    # In production, you'd fetch from database
    return {"message": "Code download endpoint", "analysis_id": analysis_id}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
