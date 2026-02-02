# AGstat Lite - Product Requirements Document

## Original Problem Statement
Build AGstat Lite — a PhD-level agricultural/horticultural biostatistics assistant with:
- Real-time AI-powered statistical analysis (OpenAI GPT-5.2 with Emergent key)
- CSV/Excel file upload for data analysis
- Real-time Python code execution for ANOVA, PCA, clustering, mixed models
- Downloadable results (plots, tables) and generated code
- Blue/black neon theme (black background, neon blue accents)
- Free to use (no authentication)

## User Personas
1. **PhD Students** - Agricultural/horticultural researchers conducting field trials
2. **Research Scientists** - Plant scientists analyzing experimental data
3. **Biostatisticians** - Professionals performing statistical consulting

## Core Requirements (Static)
- ANOVA analysis with post-hoc tests (Tukey HSD)
- PCA with biplots and variance explained
- Clustering (K-means, hierarchical)
- Assumption checks (Shapiro-Wilk, Levene's)
- Correlation matrix analysis
- Normality testing
- AI chat assistant for statistical guidance
- File upload (CSV/Excel)
- Code generation and download
- Plot generation and download

## What's Been Implemented (Feb 2, 2026)
### Backend
- FastAPI server with all statistical endpoints
- OpenAI GPT-5.2 integration via Emergent LLM key
- MongoDB for chat history storage
- File upload with pandas processing
- 6 analysis endpoints: ANOVA, PCA, Clustering, Descriptive, Correlation, Normality
- Publication-ready plot generation with neon styling
- Python code generation for each analysis

### Frontend
- React app with neon blue/black theme
- Landing page with feature highlights
- Dashboard with:
  - Chat panel (AI assistant)
  - Data preview panel
  - Analysis results panel
  - Code panel (collapsible)
- File upload modal with drag-and-drop
- Mobile responsive design
- Download functionality for code and plots

## Prioritized Backlog
### P0 (Critical) - DONE
- [x] Core statistical analyses (ANOVA, PCA, Clustering)
- [x] AI chat integration
- [x] File upload
- [x] Results visualization
- [x] Code generation

### P1 (High Priority) - Future
- [ ] Mixed models (lme4 equivalent)
- [ ] GxE analysis (AMMI, GGE biplots)
- [ ] Dose-response analysis
- [ ] Time series decomposition

### P2 (Medium Priority) - Future
- [ ] R code generation option
- [ ] LaTeX table export
- [ ] Batch analysis
- [ ] Analysis history/save

### P3 (Nice to Have) - Future
- [ ] User accounts for saved analyses
- [ ] Collaborative features
- [ ] Custom plot themes

## Next Action Items
1. Add mixed models analysis endpoint
2. Implement GxE biplot functionality
3. Add R code generation option
4. Enhance assumption check visualizations
