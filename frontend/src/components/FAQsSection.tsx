import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, HelpCircle, MessageSquare, Search, Send } from 'lucide-react';

const FAQsSection = () => {
  const [openIndex, setOpenIndex] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const faqs = [
    {
      question: "What is MolCure?",
      answer: "MolCure is an advanced platform that uses AI and machine learning to suggest effective antibiotic combinations for bacterial infections. Our technology analyzes infection characteristics to propose targeted treatment strategies."
    },
    {
      question: "How does the antibiotic suggestion system work?",
      answer: "Our system uses a sophisticated AI model trained on extensive microbiological data. When you provide details about a bacterial infection, our algorithms analyze the characteristics and suggest antibiotic combinations that are likely to be effective based on similar cases and research data."
    },
    {
      question: "Is my data kept private?",
      answer: "Yes, absolutely. We take data privacy very seriously. Any information you provide is used solely to generate personalized recommendations in real-time and is not stored or used to train our models. We adhere to strict privacy standards and security protocols."
    },
    {
      question: "How accurate are the antibiotic suggestions?",
      answer: "Our system has demonstrated a high level of accuracy in laboratory validation tests, with a success rate of approximately 94%. However, all suggestions should be reviewed by healthcare professionals before implementation, as our tool is designed to support, not replace, medical expertise."
    },
    {
      question: "Can MolCure help with antibiotic resistance?",
      answer: "Yes, addressing antibiotic resistance is one of our primary goals. By suggesting targeted combinations rather than broad-spectrum antibiotics, we aim to reduce the development of resistance. Our system also takes into account known resistance patterns when making recommendations."
    },
    {
      question: "Who can use MolCure?",
      answer: "MolCure is designed for use by healthcare professionals, researchers, and pharmaceutical companies. While our interface is user-friendly, interpreting and applying the recommendations requires professional medical or scientific knowledge."
    },
    {
      question: "Is MolCure available for commercial use?",
      answer: "Yes, we offer commercial licenses for healthcare institutions, research facilities, and pharmaceutical companies. Please contact our sales team for more information about licensing options and pricing."
    },
    {
      question: "How often is the system updated?",
      answer: "We continuously update our system with the latest research findings and data. Major updates to our algorithms and database typically occur quarterly, while minor improvements are implemented on an ongoing basis."
    }
  ];

  const filteredFaqs = faqs.filter(faq => 
    faq.question.toLowerCase().includes(searchQuery.toLowerCase()) || 
    faq.answer.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleFaq = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="bg-white min-h-screen">
      {/* Hero section */}
      <div className="relative bg-gradient-to-b from-gray-50 to-white py-16">
        <div className="absolute top-0 left-0 w-full h-64 bg-gradient-radial from-primary-200/30 to-transparent"></div>
        <div className="absolute -top-10 right-10 w-72 h-72 bg-secondary-300/20 rounded-full filter blur-3xl"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.h1 
              className="text-4xl font-extrabold sm:text-5xl sm:tracking-tight lg:text-6xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              Frequently Asked <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-secondary-600">Questions</span>
            </motion.h1>
            <motion.div 
              className="mt-6 flex justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.8 }}
            >
              <div className="h-1 w-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded"></div>
            </motion.div>
            <motion.p 
              className="mt-6 max-w-2xl mx-auto text-xl text-gray-500"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              Find answers to common questions about our platform and technology
            </motion.p>
          </motion.div>

          {/* Search bar */}
          <motion.div 
            className="max-w-2xl mx-auto mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          >
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                className="block w-full pl-10 pr-3 py-4 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500 shadow-sm"
                placeholder="Search for questions or keywords..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </motion.div>
        </div>
      </div>

      {/* FAQs list */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 pb-24">
        <div className="space-y-6">
          {filteredFaqs.length > 0 ? (
            filteredFaqs.map((faq, index) => (
              <motion.div 
                key={index}
                className="card overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <button
                  className="w-full px-6 py-5 flex justify-between items-center text-left focus:outline-none"
                  onClick={() => toggleFaq(index)}
                >
                  <div className="flex items-center">
                    <HelpCircle className="h-5 w-5 text-primary-500 mr-3 flex-shrink-0" />
                    <span className="text-lg font-medium text-gray-900">{faq.question}</span>
                  </div>
                  <div className="ml-6 flex-shrink-0">
                    {openIndex === index ? (
                      <ChevronUp className="h-5 w-5 text-gray-500" />
                    ) : (
                      <ChevronDown className="h-5 w-5 text-gray-500" />
                    )}
                  </div>
                </button>
                <AnimatePresence>
                  {openIndex === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="px-6 pb-5 text-gray-600 border-t border-gray-100 pt-3">
                        {faq.answer}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))
          ) : (
            <motion.div 
              className="text-center py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
              <p className="text-gray-500">
                We couldn't find any FAQs matching your search. Try different keywords or browse all questions.
              </p>
              <button 
                className="mt-4 btn-outline"
                onClick={() => setSearchQuery('')}
              >
                Clear search
              </button>
            </motion.div>
          )}
        </div>

        {/* Contact section */}
        <motion.div 
          className="mt-16 bg-gradient-to-r from-primary-600 to-secondary-600 rounded-2xl p-8 text-white shadow-xl"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Still have questions?</h2>
            <p className="mb-6">
              If you couldn't find the answer you were looking for, feel free to reach out to our team.
            </p>
            <motion.button 
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-primary-700 bg-white hover:bg-gray-50 transition duration-300 shadow-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Contact Us
              <Send className="ml-2 h-4 w-4" />
            </motion.button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default FAQsSection;