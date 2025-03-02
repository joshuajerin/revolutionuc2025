import React from 'react';
import { motion } from 'framer-motion';
import { Beaker, FlaskRound as Flask, Microscope, Brain, Zap, Shield, Users } from 'lucide-react';
import { Link } from 'react-router-dom';

const AboutSection = () => {
  const fadeIn = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      initial="hidden"
      animate="visible"
      transition={{ duration: 0.5 }}
      className="bg-white"
    >
      {/* Hero section */}
      <div className="relative overflow-hidden bg-gradient-to-b from-primary-50 to-white py-16">
        <div className="absolute top-0 left-0 w-full h-64 bg-gradient-radial from-primary-200/30 to-transparent"></div>
        <div className="absolute -top-10 right-10 w-72 h-72 bg-secondary-300/20 rounded-full filter blur-3xl"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            variants={fadeIn}
            transition={{ duration: 0.8 }}
          >
            <motion.h1 
              className="text-4xl font-extrabold sm:text-5xl sm:tracking-tight lg:text-6xl"
              variants={fadeIn}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              About <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">AntibiSim</span>
            </motion.h1>
            <motion.div 
              className="mt-6 flex justify-center"
              variants={fadeIn}
              transition={{ delay: 0.3, duration: 0.8 }}
            >
              <div className="h-1 w-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded"></div>
            </motion.div>
          </motion.div>

          <motion.div 
            className="prose prose-lg prose-primary mx-auto text-gray-500 max-w-4xl"
            variants={fadeIn}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <p className="text-xl leading-relaxed mb-8 text-center">
              Welcome to AntibiSim, your go-to platform for innovative antibiotic suggestions. 
              Our mission is simple: if you're dealing with a bacterial infection, just enter 
              the details, and our AI will suggest a targeted antibiotic combination to help combat it.
            </p>
          </motion.div>
        </div>
      </div>

      {/* Features section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="grid md:grid-cols-2 gap-12 my-16"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <motion.div 
              className="card p-8 border-t-4 border-primary-500"
              whileHover={{ y: -10 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <div className="flex items-center mb-6">
                <div className="bg-primary-100 p-3 rounded-full">
                  <Microscope className="h-8 w-8 text-primary-600" />
                </div>
                <h3 className="ml-4 text-xl font-bold text-gray-900">Advanced Technology</h3>
              </div>
              <p className="text-gray-600">
                Our advanced simulation technology, inspired by cutting-edge research like MIT's 
                MolecularTransformer, analyzes the specific characteristics of the infection you 
                describe to propose effective antibiotic strategies.
              </p>
            </motion.div>

            <motion.div 
              className="card p-8 border-t-4 border-secondary-500"
              whileHover={{ y: -10 }}
              transition={{ type: "spring", stiffness: 300 }}
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
          </motion.div>

          <motion.div 
            className="animated-bg rounded-3xl p-10 text-white shadow-xl mb-16 overflow-hidden relative"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            {/* Decorative elements */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2"></div>
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/10 rounded-full translate-y-1/2 -translate-x-1/2"></div>
            
            <div className="relative z-10">
              <div className="flex items-center mb-6">
                <Flask className="h-10 w-10 mr-4" />
                <h3 className="text-3xl font-bold">How It Works</h3>
              </div>
              <p className="text-xl mb-8 max-w-3xl">
                Simply enter your bacterial infection details, and let our AI 
                assist you in identifying a potentially effective antibiotic solution.
              </p>
              <motion.div 
                className="mt-6"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link 
                  to="/dashboard" 
                  className="inline-flex items-center px-8 py-4 rounded-lg text-primary-700 bg-white font-medium hover:bg-gray-50 transition duration-300 shadow-lg"
                >
                  Get Started
                  <ArrowRight className="ml-2" size={18} />
                </Link>
              </motion.div>
            </div>
          </motion.div>

          <motion.div 
            className="border-t border-gray-200 pt-12"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Our Approach</h2>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {[
                {
                  icon: <Zap className="h-6 w-6" />,
                  title: "Data-Driven",
                  description: "We analyze vast datasets of bacterial responses to identify effective treatments."
                },
                {
                  icon: <Shield className="h-6 w-6" />,
                  title: "Ethical Research",
                  description: "All our recommendations are based on ethically sourced and validated data."
                }
              ].map((item, index) => (
                <motion.div 
                  key={index}
                  className="card p-6"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1, duration: 0.5 }}
                  whileHover={{ y: -5 }}
                >
                  <div className="h-12 w-12 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center mb-4">
                    {item.icon}
                  </div>
                  <h3 className="text-xl font-medium text-gray-900 mb-2">{item.title}</h3>
                  <p className="text-gray-600">{item.description}</p>
                </motion.div>
              ))}
            </div>
            
            <div className="prose prose-lg max-w-4xl">
              <p className="mb-6 text-gray-600">
                At AntibiSim, we combine the latest in machine learning with established microbiological 
                principles to create a tool that can help address the growing challenge of antibiotic 
                resistance. By analyzing patterns in bacterial susceptibility and antibiotic efficacy, 
                our system can suggest combinations that might be overlooked in traditional approaches.
              </p>
              <p className="text-gray-600">
                While our recommendations are based on sophisticated algorithms and extensive research, 
                they should always be considered in consultation with healthcare professionals. AntibiSim 
                is designed to be a supportive tool, not a replacement for medical expertise.
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

// Import at the top was missing this component
const ArrowRight = ({ className = "", size = 24 }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
  >
    <path d="M5 12h14"></path>
    <path d="m12 5 7 7-7 7"></path>
  </svg>
);

export default AboutSection;