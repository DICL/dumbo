import React, { Component } from 'react'
// import ApplicationMonitoring from './ApplicationMonitoring'
import VCoreStatus from './YarnMonitoring/vCoreStatus'
import ReactTable from './ReactTable'
import MemoryStatus from './YarnMonitoring/MemoryStatus'
import ContainerStatus from './YarnMonitoring/ContainerStatus'
import YarnTimeHistoryView from './YarnTimeHistoryView'
import { connect } from 'react-redux';
import './AppMonitor.css';

import * as modals_actions from '../../actions/Modals';


class AppMonitor extends Component{

  viewHistory = (id,start_time) => {
    // console.log(id);
    let result = {
      id : id,
      start_time : start_time,
      size : {
        width : '843px',
        height: 'auto'
      }
    }
    this.props.create_modal(result,'ApplicationHistory');
  }

  viewTimeHistory = (result) => {
    this.props.create_modal(result,'YarnJobHistoryPerNode');
  }

  render () {
    return (
      <div>
        <div>
          <div className="right_wrap pull-left" >
            <div className="title_wrap">
              <div className="text">Application Monitoring</div>
            </div>
            <div className="contents_wrap">
              <div id="YarnMonitoring" className="cluster_info">
                {/*
                  <ApplicationMonitoring/>
                  */}
                  <div className="graph_area">
                    <VCoreStatus yarn_status={this.props.yarn_status} />
                  </div>
                  <div className="graph_area">
                    <MemoryStatus yarn_status={this.props.yarn_status} />
                  </div>
                  <div className="graph_area">
                    <ContainerStatus yarn_status={this.props.yarn_status}/>
                  </div>



              </div>
            </div>
          </div>

          <div className="right_wrap pull-left">
            <div className="title_wrap">
              <div className="text">YARN Application History</div>
            </div>
            <div className="contents_wrap">
                <div className="cluster_info" id="YarnTimeHistoryView">
                  <YarnTimeHistoryView
                    viewTimeHistory={this.viewTimeHistory}
                    />
                </div>
                <div className="cluster_info" id="ReactTable">
                  <ReactTable
                    yarn_job_list={this.props.yarn_job_list}
                    viewHistory={this.viewHistory}
                    />
                </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}


const mapStateToProps = (state) => {
    return {
        yarn_job_list: state.status.yarn_job_list,
        yarn_status: state.status.yarn_status,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
      create_modal: (modal_data,component_name) => {
        //console.log(modal_data);
        dispatch(modals_actions.create_modal(modal_data,component_name))
      },
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(AppMonitor)
