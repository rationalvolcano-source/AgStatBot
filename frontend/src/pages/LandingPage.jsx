import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  LineChart, 
  PieChart, 
  TrendingUp, 
  FlaskConical, 
  Dna, 
  ArrowRight,
  Upload,
  Code,
  Download
} from 'lucide-react';
import { Button } from '@/components/ui/button';

const features = [
  {
    icon: BarChart3,
    title: 'ANOVA & Post-Hoc',
    description: 'RCBD, split-plot, factorial designs with Tukey HSD comparisons'
  },
  {
    icon: PieChart,
    title: 'PCA & Clustering',
    description: 'Principal Component Analysis with biplots, K-means & hierarchical clustering'
  },
  {
    icon: LineChart,
    title: 'Mixed Models',
    description: 'Random effects, repeated measures, and GxE interaction analysis'
  },
  {
    icon: TrendingUp,
    title: 'Dose-Response',
    description: 'Log-logistic, probit models for pesticide and treatment efficacy'
  },
  {
    icon: FlaskConical,
    title: 'Assumption Checks',
    description: 'Shapiro-Wilk normality, Levene homogeneity, residual diagnostics'
  },
  {
    icon: Dna,
    title: 'GxE Analysis',
    description: 'AMMI, GGE biplots for multi-environment trial analysis'
  }
];

const stats = [
  { value: '50+', label: 'Statistical Methods' },
  { value: 'Real-time', label: 'Code Execution' },
  { value: 'Free', label: 'No Account Needed' }
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#020408] overflow-hidden">
      {/* Grid Background */}
      <div className="fixed inset-0 grid-bg opacity-50 pointer-events-none" />
      
      {/* Hero Section */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6 py-20">
        {/* Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#020408]/50 to-[#020408] pointer-events-none" />
        
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="relative z-10 text-center max-w-4xl mx-auto"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-full bg-[#0B1121] border border-[#1E293B] text-sm"
          >
            <span className="w-2 h-2 rounded-full bg-[#00F0FF] animate-pulse" />
            <span className="text-[#94A3B8]">PhD-Level Biostatistics</span>
          </motion.div>

          {/* Title */}
          <h1 
            data-testid="hero-title"
            className="font-['Chivo'] font-black text-5xl md:text-7xl tracking-tighter mb-6"
          >
            <span className="text-[#F8FAFC]">AG</span>
            <span className="text-[#00F0FF] neon-text">stat</span>
            <span className="text-[#F8FAFC]"> Lite</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-[#94A3B8] mb-4 max-w-2xl mx-auto">
            Your AI-powered agricultural biostatistics assistant.
          </p>
          <p className="text-base text-[#64748B] mb-10 max-w-xl mx-auto italic">
            "Congratulations on surviving field trials, Excel disasters, and questionable CSV exports..."
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button
              data-testid="get-started-btn"
              onClick={() => navigate('/dashboard')}
              className="bg-[#00F0FF] text-black hover:bg-[#00CCFF] font-bold rounded-sm shadow-[0_0_15px_rgba(0,240,255,0.3)] hover:shadow-[0_0_25px_rgba(0,240,255,0.5)] transition-all uppercase tracking-wider text-sm px-8 py-6"
            >
              Start Analyzing
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button
              data-testid="learn-more-btn"
              variant="outline"
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              className="bg-transparent border-[#1E293B] text-[#94A3B8] hover:border-[#00F0FF] hover:text-[#00F0FF] rounded-sm uppercase tracking-wider text-sm px-8 py-6"
            >
              Learn More
            </Button>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="relative z-10 mt-20 grid grid-cols-3 gap-8 max-w-2xl mx-auto"
        >
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl md:text-4xl font-['Chivo'] font-black text-[#00F0FF] mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-[#64748B] uppercase tracking-wider">
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-[#1E293B] rounded-full flex justify-center p-2">
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-1 h-2 bg-[#00F0FF] rounded-full"
            />
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-['Chivo'] font-black text-3xl md:text-4xl tracking-tight text-[#F8FAFC] mb-4">
              Statistical <span className="text-[#00F0FF]">Arsenal</span>
            </h2>
            <p className="text-[#94A3B8] max-w-xl mx-auto">
              Publication-ready analyses for agricultural and horticultural research
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                data-testid={`feature-card-${index}`}
                className="group p-6 bg-[#0B1121]/60 border border-[#1E293B] rounded-sm hover:border-[#00F0FF]/50 transition-all duration-300"
              >
                <div className="w-12 h-12 mb-4 flex items-center justify-center rounded-sm bg-[#1E293B] group-hover:bg-[#00F0FF]/10 transition-colors">
                  <feature.icon className="w-6 h-6 text-[#00F0FF]" />
                </div>
                <h3 className="font-['Chivo'] font-bold text-lg text-[#F8FAFC] mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-[#94A3B8] leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative py-24 px-6 bg-[#0B1121]/30">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="font-['Chivo'] font-black text-3xl md:text-4xl tracking-tight text-[#F8FAFC] mb-4">
              How It <span className="text-[#00F0FF]">Works</span>
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: Upload, title: 'Upload Data', desc: 'Drop your CSV or Excel file' },
              { icon: Code, title: 'AI Analysis', desc: 'Get code & results instantly' },
              { icon: Download, title: 'Export', desc: 'Download plots & scripts' }
            ].map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="text-center"
              >
                <div className="relative inline-flex items-center justify-center w-16 h-16 mb-6 rounded-full bg-[#1E293B] border-2 border-[#00F0FF]/30">
                  <step.icon className="w-7 h-7 text-[#00F0FF]" />
                  <span className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-[#00F0FF] text-black font-bold text-sm flex items-center justify-center">
                    {index + 1}
                  </span>
                </div>
                <h3 className="font-['Chivo'] font-bold text-lg text-[#F8FAFC] mb-2">
                  {step.title}
                </h3>
                <p className="text-sm text-[#94A3B8]">
                  {step.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="relative py-24 px-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto text-center"
        >
          <h2 className="font-['Chivo'] font-black text-3xl md:text-4xl tracking-tight text-[#F8FAFC] mb-6">
            Ready to <span className="text-[#00F0FF]">Analyze</span>?
          </h2>
          <p className="text-[#94A3B8] mb-8">
            No sign-up required. Start your statistical analysis now.
          </p>
          <Button
            data-testid="cta-btn"
            onClick={() => navigate('/dashboard')}
            className="bg-[#00F0FF] text-black hover:bg-[#00CCFF] font-bold rounded-sm shadow-[0_0_15px_rgba(0,240,255,0.3)] hover:shadow-[0_0_25px_rgba(0,240,255,0.5)] transition-all uppercase tracking-wider text-sm px-10 py-6"
          >
            Launch Dashboard
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[#1E293B] py-8 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="font-['Chivo'] font-bold text-lg">
            <span className="text-[#F8FAFC]">AG</span>
            <span className="text-[#00F0FF]">stat</span>
            <span className="text-[#F8FAFC]"> Lite</span>
          </div>
          <p className="text-sm text-[#64748B]">
            PhD-Level Agricultural Biostatistics
          </p>
        </div>
      </footer>
    </div>
  );
}
