import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import About from './components/About';
import Contact from './components/Contact'
import './App.css'; 
import Home from './components/Home'

function App() {
  return (
    <div>
      <Home />
    </div>
  );
}

export default App;
