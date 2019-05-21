import React from 'react';
import './YarnJobHistoryPerNode.css';
import GetYarnJobHistoryPerNode from './GetYarnJobHistoryPerNode';

// idx : modal 창 고유키, onRemove : modal 창 닫는 메서드, data : 기타 데이터
const YarnJobHistoryPerNode = ({idx,  onRemove, data  }) => {
  return(
    <div className="YarnJobHistoryPerNode ui card handle" style={{}}>
      <div className='content'>
        <div className='header cursor'>
            <span>
                <span>{data.node}</span>
                <span style={{ marginLeft : '10px' }}>Yarn Job History View</span>
            </span>

            <button
              className="popup-remove"
              onClick={ (e) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  onRemove(idx);  }  }
              >
                &times;
            </button>
        </div>
      </div>
      <div className='content body'>
        <div className='description'>

          <div>
            <GetYarnJobHistoryPerNode
              data={data}
              idx={idx}
              onRemove={onRemove}
              />


          </div>
        </div>
      </div>

    </div>
 );
}

export default YarnJobHistoryPerNode;
