import React from 'react';
import './vconsole.css';

import Xtrem from './xterm';

const Vconsole = ({idx,  onRemove, data}) => {
    return (
          <div className='ui card Vconsole'>
            <div className='content'>
              <div className='header'>
                <span>{data} console</span>
                <button className="popup-remove" onClick={ (e) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  onRemove(idx);  }  } >&times;</button>
              </div>
            </div>
            <div className='content console'>
              <div className='description'>

                  <Xtrem
                    host_name={data}
                    onRemove={onRemove}
                    idx={idx}
                    />

              </div>
            </div>
            <div className='extra content'>
            </div>
          </div>
    );
}

export default Vconsole;
