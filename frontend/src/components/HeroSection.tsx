import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Zap, Shield, Microscope, Brain, FlaskConical } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const HeroSection = () => {
  const navigate = useNavigate();

  return (
    <div className="relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary-300/30 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-80 h-80 bg-secondary-300/20 rounded-full filter blur-3xl"></div>
      </div>

      {/* Hero content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight mb-4">
            <span className="block text-gray-900">Welcome to</span>
            <span className="block mt-1 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">
              MolCure
            </span>
          </h1>
          <p className="mt-6 text-lg text-gray-600 max-w-2xl mx-auto">
            Your go-to platform for innovative antibiotic suggestions. Enter your bacterial infection details, 
            and our AI will suggest targeted antibiotic combinations to help combat it.
          </p>
          <motion.div 
            className="flex justify-center mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <div className="h-1 w-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded"></div>
          </motion.div>
          <motion.button 
            className="btn-primary group mt-8"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate('/dashboard')}
          >
            Start Simulation
            <ArrowRight className="ml-2 group-hover:translate-x-1 transition-transform" size={18} />
          </motion.button>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-12 my-16">
          <motion.div 
            className="card p-8 border-t-4 border-primary-500"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            whileHover={{ y: -10 }}
          >
            <div className="flex items-center mb-6">
              <div className="bg-primary-100 p-3 rounded-full">
                <Microscope className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="ml-4 text-xl font-bold text-gray-900">Advanced Technology</h3>
            </div>
            <p className="text-gray-600">
              Our advanced simulation technology analyzes the specific characteristics of the infection you 
              describe to propose effective antibiotic strategies.
            </p>
          </motion.div>

          <motion.div 
            className="card p-8 border-t-4 border-secondary-500"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            whileHover={{ y: -10 }}
          >
            <div className="flex items-center mb-6">
              <div className="bg-secondary-100 p-3 rounded-full">
                <Brain className="h-8 w-8 text-secondary-600" />
              </div>
              <h3 className="ml-4 text-xl font-bold text-gray-900">AI-Powered Solutions</h3>
            </div>
            <p className="text-gray-600">
              Rest assured, your input is used solely to generate a personalized recommendation 
              in real time and is not employed for training our underlying model.
            </p>
          </motion.div>
        </div>

        {/* How It Works Section */}
        <motion.div 
          className="animated-bg rounded-3xl p-10 text-white shadow-xl mb-16 overflow-hidden relative"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/10 rounded-full translate-y-1/2 -translate-x-1/2"></div>
          
          <div className="relative z-10">
            <div className="flex items-center mb-6">
              <FlaskConical className="h-10 w-10 mr-4" />
              <h3 className="text-3xl font-bold">Our Approach</h3>
            </div>
            <div className="grid md:grid-cols-2 gap-8 mt-8">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <Zap className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4">
                  <h4 className="text-xl font-semibold mb-2">Data-Driven</h4>
                  <p className="text-white/80">
                    We analyze vast datasets of bacterial responses to identify effective treatments.
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <Shield className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4">
                  <h4 className="text-xl font-semibold mb-2">Ethical Research</h4>
                  <p className="text-white/80">
                    All our recommendations are based on ethically sourced and validated data.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default HeroSection;