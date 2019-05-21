import React, { Component } from 'react'
import './Status.css';
class Status extends Component {

  calPersente = (data,total) => {
    return data/total * 100 + "%";
  }

  calReversePersente = (data,total) => {
    return (total-data)/total * 100 + "%";
  }

  componentDidMount(){
  }

  render() {

    const host_starus = this.props.host_starus;
    const storage_usage = {
      "CapacityUsed" :  (this.props.storage_usage.CapacityUsed / Math.pow(1024, 4)).toFixed(2),
      "CapacityTotal" : (this.props.storage_usage.CapacityTotal / Math.pow(1024, 4)).toFixed(2),
      "CapacityRemaining" : (this.props.storage_usage.CapacityRemaining / Math.pow(1024, 4)).toFixed(2),
    }
    const yarn_job_status = this.props.yarn_job_status;
    const yarn_status = this.props.yarn_status;
    //console.log(yarn_status);

    return(
      <div>
        <div className="graph_wrap">
          <div className="bullet">  </div>
          <div className="graph_title">Live Status</div>
          <div className="clearfix"></div>
          <div id="liveStatusBar">
            <div className="graph_label">
              <div className="graph_label">
                <span className="value">{host_starus.started_count}</span>
                <span className="total">/ {host_starus.total_count} Nodes</span>
              </div>
            </div>
            <div className="bar_graph">
              <div className="value1" style={{ 'width' : this.calPersente(host_starus.started_count,host_starus.total_count) }}></div>
              <div className="value2" style={{ 'width' : this.calReversePersente(host_starus.started_count ,host_starus.total_count) }}></div>
              <div className="clearfix"></div>
            </div>
            <div className="bar_legend_wrap">
                <span className="rect value1"></span>
                <span className="text">Live</span>
                <span className="rect value2"></span>
                <span className="text">Dead</span>
            </div>
          </div>
        </div>

        <div className="graph_wrap">
          <div className="bullet"></div>
          <div className="graph_title">Shared Storage Usage Status</div>
          <div className="clearfix"></div>
          <div id="storageUsageBar">
            <div className="graph_label">
              <span className="value">{storage_usage.CapacityUsed}</span><span className="total"> / {storage_usage.CapacityTotal} TB</span>
            </div>
            <div className="bar_graph">
              <div className="value1" style={{'width' : this.calPersente(storage_usage.CapacityUsed,storage_usage.CapacityTotal)}}></div>
              <div className="value2" style={{'width' : this.calReversePersente(storage_usage.CapacityUsed,storage_usage.CapacityTotal)}}></div>
              <div className="clearfix"></div>
            </div>
            <div className="bar_legend_wrap">
                <span className="rect value1"></span>
                <span className="text">Used</span>
                <span className="rect value2"></span>
                <span className="text">Remaining</span>
            </div>
          </div>
        </div>


        <div className="graph_wrap">
          <div className="bullet"></div>
          <div className="graph_title">YARN Jobs Status</div>
          <div className="clearfix"></div>
          <div id="storageUsageBar">
            <div className="text_graph">
              <p><span className="value">{yarn_job_status.running}</span> Running</p>
              <p><span className="value">{yarn_job_status.completed}</span> Completed</p>
              <p><span className="value">{yarn_job_status.failed}</span> Failed</p>
            </div>
          </div>
        </div>

        <div className="graph_wrap">
          <div className="bullet"></div>
          <div className="graph_title">YARN vCore Status</div>
          <div className="clearfix"></div>
          <div id="yarnVcoreStatus">
            <div className="graph_label"><span className="value">{yarn_status.allocatedVirtualCores}</span><span className="total"> / {yarn_status.totalVirtualCores} Core</span></div>
            <div className="bar_graph">
              <div className="value1" style={{ 'width' : this.calPersente(yarn_status.allocatedVirtualCores,yarn_status.totalVirtualCores) }}></div>
              <div className="value2" style={{ 'width' : this.calReversePersente(yarn_status.allocatedVirtualCores,yarn_status.totalVirtualCores) }}></div>
              <div className="clearfix"></div>
            </div>
            <div className="bar_legend_wrap">
                <span className="rect value1" ></span><span className="text">Used</span>
                <span className="rect value2" ></span><span className="text">Remaining</span>
            </div>
          </div>
        </div>

        <div className="graph_wrap">
          <div className="bullet"></div>
          <div className="graph_title">Yarn Memory Status</div> <div className="clearfix"></div>
          <div id="yarnMemoryBar">
            <div className="graph_label"><span className="value">{ (yarn_status.allocatedMB / 1024).toFixed(2)}</span><span className="total"> / { (yarn_status.totalMB / 1024).toFixed(2) } GB</span></div>
            <div className="bar_graph">
              <div className="value1" style={{ 'width' : this.calPersente(yarn_status.allocatedMB,yarn_status.totalMB) }}></div>
              <div className="value2" style={{ 'width' : this.calReversePersente(yarn_status.allocatedMB,yarn_status.totalMB) }}></div>
              <div className="clearfix"></div>
            </div>
            <div className="bar_legend_wrap">
                <span className="rect value1"></span><span className="text">Used</span>
                <span className="rect value2"></span><span className="text">Remaining</span>
            </div>
          </div>
        </div>
      </div>
    )}

}
export default Status;
