import React from 'react';
import Logo from './Logo/Logo';
import Navbar from './Navbar/Navbar';
import './Header.css';


const Header = () => (
    <div className="Header">
        {/* Header */}
        <Logo/>
        <Navbar/>

          {/*<Menu/>*/}
    </div>
)

export default Header;
