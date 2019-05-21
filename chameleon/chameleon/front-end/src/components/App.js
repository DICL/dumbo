import React from 'react';
import { Route } from 'react-router-dom';
import { Home ,Yarn ,Lustre} from '../components/Pages/index.async.js';
// Warning: Can't call setState (or forceUpdate) on an unmounted component 에러발생으로 index.async.js 헤제
// import Home from './Pages/Home/Home.js'
// import Yarn from './Pages/Yarn/Yarn.js'
// import Lustre from './Pages/Lustre/Lustre.js'

import Header from '../components/Header/Header';
import Menu from '../components/Menu/Menu';
import { HashRouter } from 'react-router-dom';
import './App.css';



const App = () => {
  return (
    <HashRouter>
      <div className="wrapper">
          <Header/>
          <div className="ui container" >
            <Menu/>
            <div className="right_area" >
              <Route exact path="/" component={Home}/>
              <Route path="/yarn" component={Yarn}/>
              <Route path="/lustre" component={Lustre}/>
            </div>
          </div>
      </div>
    </HashRouter>
  );
};

export default App;
