import React from 'react';
import './LustreFS.css';
import MDS from './MDS';
import OSS from './OSS';

const LustreFS = ({mds_metric,oss_metric}) => (
  <div className="LustreFS">
    <div className="right_wrap pull-left" index="1">
      <div className="title_wrap">
        <div className="text">MDS</div>
      </div>
      <div className="cluster_wrap">
        <div className="graph_wp" >
          <MDS
            mds_metric={mds_metric}
            />
        </div>
      </div>
    </div>
    <div className="right_wrap pull-left" index="3">
      <div className="title_wrap">
        <div className="text">
          OSS
        </div>
      </div>
      <div id="lustre" className="cluster_wrap" >
        <div className="graph_wp" >
          <OSS
            oss_metric={oss_metric}
            />
        </div>
      </div>
    </div>
  </div>
)


export default LustreFS;
