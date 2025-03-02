import React from 'react';
import { Beaker, Mail, Github, Linkedin, ArrowUpRight, Phone } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

interface SocialLinkProps {
  href: string;
  icon: React.ReactNode;
  label: string;
}

interface FooterLinkProps {
  to: string;
  text: string;
}

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 z-0 opacity-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary-500 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-80 h-80 bg-secondary-500 rounded-full filter blur-3xl"></div>
      </div>
      
      <div className="relative z-10 max-w-7xl mx-auto py-8 px-4 sm:px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Logo and Social Links */}
          <div className="flex flex-col items-center md:items-start">
            <motion.div 
              className="flex items-center"
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <div className="relative">
                <div className="absolute inset-0 bg-primary-500 rounded-full blur-md opacity-30"></div>
                <Beaker className="h-8 w-8 text-primary-400 relative z-10" />
              </div>
              <span className="ml-2 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-400 to-secondary-400">MolCure</span>
            </motion.div>
            <div className="flex space-x-6 mt-4">
              <SocialLink href="#" icon={<Github className="h-5 w-5" />} label="GitHub" />
              <SocialLink href="#" icon={<Linkedin className="h-5 w-5" />} label="LinkedIn" />
            </div>
          </div>

          {/* Quick Links */}
          <div className="text-center md:text-left">
            <h3 className="text-sm font-semibold text-gray-300 tracking-wider uppercase mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <FooterLink to="/" text="Home" />
              <FooterLink to="/dashboard" text="Dashboard" />
            </ul>
          </div>

          {/* Contact */}
          <div className="text-center md:text-left">
            <h3 className="text-sm font-semibold text-gray-300 tracking-wider uppercase mb-4">Contact</h3>
            <div className="space-y-2">
              <a href="mailto:info@MolCure.com" className="text-gray-400 hover:text-white transition-colors flex items-center justify-center md:justify-start">
                <Mail className="h-4 w-4 mr-2" />
                info@MolCure.com
              </a>
              <a href="tel:+1234567890" className="text-gray-400 hover:text-white transition-colors flex items-center justify-center md:justify-start">
                <Phone className="h-4 w-4 mr-2" />
                +1 (234) 567-890
              </a>
            </div>
          </div>
        </div>
        
        <div className="mt-8 pt-4 border-t border-gray-800">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm">
            <p className="text-gray-400">
              &copy; 2025 MolCure. All rights reserved.
            </p>
            <div className="mt-2 md:mt-0 flex space-x-4">
              <a href="#" className="text-gray-400 hover:text-white transition-colors">Privacy</a>
              <a href="#" className="text-gray-400 hover:text-white transition-colors">Terms</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

const SocialLink = ({ href, icon, label }: SocialLinkProps) => (
  <motion.a 
    href={href} 
    className="text-gray-400 hover:text-white transition-colors"
    whileHover={{ scale: 1.2, rotate: 5 }}
    whileTap={{ scale: 0.9 }}
  >
    <span className="sr-only">{label}</span>
    {icon}
  </motion.a>
);

const FooterLink = ({ to, text }: FooterLinkProps) => (
  <li>
    <Link 
      to={to} 
      className="text-base text-gray-400 hover:text-white transition-colors flex items-center justify-center md:justify-start group"
    >
      {text}
      <ArrowUpRight 
        size={14} 
        className="ml-1 opacity-0 group-hover:opacity-100 transition-opacity" 
      />
    </Link>
  </li>
);

export default Footer;