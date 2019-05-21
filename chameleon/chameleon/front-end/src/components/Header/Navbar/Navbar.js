import React, { Component } from 'react';
import './Navbar.css';
import Alert from './Alert';

// import icon_alerts from './img/icon_alerts.png';
// import icon_setting from './img/icon_setting.png';
// import icon_refresh from './img/icon_refresh.png';
// import icon_logout from './img/icon_logout.png';

class Navbar extends Component{



  render() {
    return(
      <nav className="navbar navbar-default navbar-static-top" id="nav">
      <div className="ui container" >
       <div id="navbar" className="navbar-collapse collapse">
         <ul className="nav navbar-nav pull-left">
           <li className=""><a href="#/" menu="main" >Summary</a></li>
           <li className=""><a href="#/yarn" menu="yarn" >YARN application monitor</a></li>
           <li className=""><a href="#/lustre" menu="lustre">HPC resource monitor</a></li>
           <li className=""><a href="http://192.168.1.191:8080/" menu="ambari" >Ambari</a></li>
         </ul>
          <ul className="nav navbar-nav pull-right">
           <Alert/>
           {/*
             <li className="nav_icon setting"><img src={icon_setting} alt="setting"/></li>
             <li className="nav_icon refresh"><img src={icon_refresh} alt="refresh"/></li>
             <li className="nav_icon logout"><img src={icon_logout} alt="logout"/></li>
           */}
         </ul>
         <div className="clearfix"></div>
       </div>
     </div>


   </nav>
   );
  }

}

export default Navbar;
