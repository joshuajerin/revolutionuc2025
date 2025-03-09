import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Beaker, X } from 'lucide-react';

const Dashboard = () => {
  const [prompt, setPrompt] = useState('');
  const [expandedBlock, setExpandedBlock] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({ input_text: prompt.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to run simulation');
      }

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data.output);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Simulation error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const infoBlocks = [
    {
      title: "Bacterial Analysis",
      description: "Our advanced machine learning algorithms analyze bacterial strains with unprecedented precision. By leveraging extensive databases of bacterial behaviors and resistance patterns, we can identify specific strain characteristics and predict their responses to various antibiotics. This system considers multiple factors including genetic markers, environmental conditions, and historical treatment outcomes to provide comprehensive analysis results that guide effective treatment strategies.",
      image: "https://images.unsplash.com/photo-1583912267550-d44c9c34c23c?auto=format&fit=crop&w=500&h=300&q=80"
    },
    {
      title: "Treatment Safety",
      description: "Safety is paramount in our analysis process. Our comprehensive evaluation system assesses potential side effects, drug interactions, and patient-specific risk factors. We analyze historical safety data, known contraindications, and individual patient profiles to ensure recommended treatments minimize risks while maximizing effectiveness. This includes detailed analysis of drug-drug interactions, allergic reaction possibilities, and cumulative toxicity considerations.",
      image: "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=500&h=300&q=80"
    },
    {
      title: "Quick Results",
      description: "Our high-performance computing infrastructure delivers simulation results within seconds. This rapid analysis capability enables healthcare providers to evaluate multiple treatment options quickly and make time-critical decisions. The system processes vast amounts of data, running complex simulations and analyzing multiple scenarios simultaneously to provide comprehensive yet swift results that can be immediately acted upon in clinical settings.",
      image: "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=500&h=300&q=80"
    },
    {
      title: "Smart Predictions",
      description: "Our AI-driven prediction system is built on extensive medical research and clinical data analysis. By processing millions of data points from real-world clinical outcomes, research papers, and treatment histories, we generate highly accurate predictions for treatment effectiveness. The system continuously learns from new data, improving its predictive capabilities and adapting to emerging bacterial resistance patterns in real-time.",
      image: "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=500&h=300&q=80"
    },
    {
      title: "Precision Targeting",
      description: "Our precision targeting system identifies specific antibiotic combinations optimized for resistant strains. Using advanced molecular modeling and bacterial genome analysis, we can pinpoint vulnerabilities in even the most resistant bacteria. This targeted approach helps develop customized treatment strategies that are more likely to succeed where conventional approaches might fail, particularly in cases involving multi-drug resistant organisms.",
      image: "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=500&h=300&q=80"
    },
    {
      title: "Lab Validated",
      description: "Every aspect of our system has been rigorously validated through comprehensive laboratory testing and clinical trials. Our recommendations are backed by extensive experimental data and real-world clinical outcomes. We maintain partnerships with leading research institutions and hospitals to continuously validate and improve our prediction models, ensuring that our suggestions are both scientifically sound and clinically relevant for real-world applications.",
      image: "https://images.unsplash.com/photo-1583912267550-d44c9c34c23c?auto=format&fit=crop&w=500&h=300&q=80"
    }
  ];

  return (
    <div className="relative min-h-screen bg-gradient-to-b from-primary-50 to-white">
      {/* Background decorative elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary-300/30 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-80 h-80 bg-secondary-300/20 rounded-full filter blur-3xl"></div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="relative z-10 container mx-auto px-4 py-12"
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-center mb-12"
        >
          <motion.h1 
            className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight mb-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.8 }}
          >
            <span className="block text-gray-900">Welcome to</span>
            <span className="block mt-1 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">
              Simulation Dashboard
            </span>
          </motion.h1>
          <motion.div 
            className="mt-6 flex justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <div className="h-1 w-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded"></div>
          </motion.div>
          <motion.p 
            className="mt-6 text-lg text-gray-600 max-w-2xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          >
            Enter your bacterial infection details below, and our AI will suggest effective antibiotic combinations.
          </motion.p>
        </motion.div>
        

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="max-w-3xl mx-auto mb-16"
        >
          <div className="bg-white rounded-xl shadow-lg p-8 backdrop-blur-lg bg-white/80">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="prompt" className="block text-lg font-medium text-gray-700 mb-3">
                  Enter your simulation prompt
                </label>
                <div className="relative">
                  <textarea
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 min-h-[160px] text-base"
                    placeholder="Enter the SMILES for the bacterial cell wall component..."
                  />
                </div>
                <p className="mt-2 text-sm text-gray-500">
                  Include the SMILES string for the bacterial cell wall component.
                </p>
              </div>


              {result && (
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <pre className="whitespace-pre-wrap text-sm text-gray-800">{result}</pre>
                  </div>
                  
                  {/* Molecule Visualizations */}
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">Molecule Visualization</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {/* 2D Molecule Image */}
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center justify-between">
                          <span>2D Structure</span>
                          <span className="text-sm text-gray-500">(Click to enlarge)</span>
                        </h3>
                        <div 
                          className="relative w-full h-[400px] flex items-center justify-center bg-white rounded-lg border-2 border-dashed border-gray-200 overflow-hidden"
                          onClick={() => {
                            window.open('http://localhost:8000/molecule_2d', '_blank');
                          }}
                        >
                          <img
                            src="http://localhost:8000/molecule_2d"
                            alt="2D Molecule Structure"
                            className="max-w-full max-h-full object-contain hover:scale-105 transition-transform cursor-pointer"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.onerror = null;
                              target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTEyIDZWNE0xMiAyMHYtMk0xNiAxMmgyTTYgMTJINE0xNy42NiA2LjM0bC0xLjQxIDEuNDFNNy43NSAxNi4yNWwtMS40MSAxLjQxTTE3LjY2IDE3LjY2bC0xLjQxLTEuNDFNNy43NSA3Ljc1TDYuMzQgNi4zNCIgc3Ryb2tlPSIjNkI3MjgwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPjwvc3ZnPg==';
                              target.parentElement?.classList.add('animate-pulse');
                            }}
                          />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-gray-400 text-center hidden">
                              <p className="text-lg font-medium">Loading 2D Structure...</p>
                              <p className="text-sm">Please wait while we generate the visualization</p>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* 3D Molecule Visualization */}
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center justify-between">
                          <span>3D Structure</span>
                          <span className="text-sm text-gray-500">(Interactive)</span>
                        </h3>
                        <div className="relative w-full h-[400px] bg-white rounded-lg border-2 border-dashed border-gray-200 overflow-hidden">
                          <iframe
                            src="http://localhost:8000/molecule_3d"
                            className="absolute inset-0 w-full h-full"
                            frameBorder="0"
                            title="3D Molecule Visualization"
                            sandbox="allow-scripts allow-same-origin"
                            onLoad={(e) => {
                              const frame = e.target as HTMLIFrameElement;
                              frame.parentElement?.classList.remove('animate-pulse');
                            }}
                            onError={(e) => {
                              const frame = e.target as HTMLIFrameElement;
                              frame.style.display = 'none';
                              frame.parentElement?.classList.add('flex', 'items-center', 'justify-center', 'text-gray-400');
                              frame.parentElement!.innerHTML = `
                                <div class="text-center">
                                  <p class="text-lg font-medium">3D Visualization Unavailable</p>
                                  <p class="text-sm">Please try refreshing the page</p>
                                </div>
                              `;
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={isLoading}
                className={`w-full bg-gradient-to-r from-primary-600 to-secondary-600 text-white py-3 px-6 rounded-lg font-medium shadow-md hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2 ${
                  isLoading ? 'opacity-75 cursor-not-allowed' : ''
                }`}
              >
                <span>{isLoading ? 'Running Simulation...' : 'Run Simulation'}</span>
                <Beaker className="h-5 w-5" />
              </motion.button>
            </form>
          </div>
        </motion.div>

        {/* Info Blocks */}
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {infoBlocks.map((block, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 + index * 0.1 }}
                onClick={() => setExpandedBlock(index)}
                className="bg-white rounded-xl shadow-md hover:shadow-lg overflow-hidden cursor-pointer h-[250px]"
              >
                <div className="flex h-full">
                  <div className="w-1/2">
                    <img
                      src={block.image}
                      alt={block.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="w-1/2 p-6 flex flex-col">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">
                      {block.title}
                    </h3>
                    <div className="text-gray-600 overflow-hidden flex-1">
                      <div className="pr-4">
                        <p className="line-clamp-2">{block.description}</p>
                        <div className="flex items-center mt-1">
                          <span className="text-gray-600 text-sm">...</span>
                          <span className="text-primary-600 font-medium text-sm ml-1">Click to read more</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Popup Overlay */}
        <AnimatePresence>
          {expandedBlock !== null && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
                onClick={() => setExpandedBlock(null)}
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ type: "spring", duration: 0.5 }}
                  className="relative w-full max-w-4xl bg-white rounded-xl shadow-2xl overflow-hidden"
                  style={{ height: '600px' }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    onClick={() => setExpandedBlock(null)}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 transition-colors z-10 bg-white/80 backdrop-blur-sm"
                  >
                    <X className="w-6 h-6 text-gray-600" />
                  </button>
                  <div className="flex h-full">
                    <div className="w-[55%] h-full">
                      <img
                        src={infoBlocks[expandedBlock].image}
                        alt={infoBlocks[expandedBlock].title}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="w-[45%] p-8 overflow-y-auto">
                      <h2 className="text-2xl font-semibold text-gray-900 mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">
                        {infoBlocks[expandedBlock].title}
                      </h2>
                      <p className="text-gray-600 text-lg leading-relaxed">
                        {infoBlocks[expandedBlock].description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default Dashboard; 