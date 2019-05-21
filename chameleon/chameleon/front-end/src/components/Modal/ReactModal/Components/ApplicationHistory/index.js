import React from 'react';
import YarnJobHistory from './getYarnJobHistory'
import './ApplicationHistory.css';


const ApplicationHistory = ({idx,  onRemove, data}) => {
    return (
          <div className='ui card ApplicationHistory'>
            <div className='content'>
              <div className='header cursor'>
                <span>{data.id}</span>
                <button className="popup-remove" onClick={ (e) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  onRemove(idx);  }  } >&times;</button>
              </div>
            </div>
            <div className='content console'>
              <div className='description'>
                <YarnJobHistory
                  application_id={data.id}
                  start_time={data.start_time}
                  idx={idx}
                  onRemove={onRemove}
                  />
              </div>
            </div>
            <div className='extra content'>
            </div>
          </div>
    );
}

export default ApplicationHistory;
