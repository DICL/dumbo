import React from 'react';
import { NavLink } from 'react-router-dom';
import './Logo.css';
import LogoImg from './logo.png';


const Logo = () => (

    <header id="header">
      <div className="ui container">
        <div className="logo"> <NavLink exact to="/"><img src={LogoImg} alt="logo" />Chameleon</NavLink> </div>
      </div>
    </header>
)

export default Logo;
