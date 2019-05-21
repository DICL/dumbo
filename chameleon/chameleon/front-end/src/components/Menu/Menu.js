import React, { Component } from 'react'
// import { NavLink } from 'react-router-dom';
import Status from './Status/Status.js';
import * as service from '../../services/getStatus';
import { connect } from 'react-redux';
import * as actions from '../../actions/Status';
import _ from 'lodash';



class Menu extends Component {
  constructor(props) {
      super(props);
      this.state = {
        "yarn_job_status" : {
          "running" : 0,
          "completed" : 0,
          "failed" : 0,
        },
        "server_status_intervalId": null,
     };

  }


  getServerState = async () => {
    const host_starus = await service.getHostStatus();
    const storage_usage = await service.getStorageUsage();
    const update_yarn_job_list = await service.getYarnAppList();
    const yarn_status = await service.getYarnStatus();

    this.props.update_host_status(
      host_starus.data
    );

    this.props.update_storage_usage(
      storage_usage.data
    );

    this.props.update_yarn_job_list(
      update_yarn_job_list.data.apps.app
    );

    this.props.update_yarn_status(
      yarn_status.data.clusterMetrics
    );

    // yarn job list 에서 RUNNING FINISHED FAILED 상태인것들만 추출하여 갯수을 파악한다
    const yarn_job_status = {
      "running" : _.filter(update_yarn_job_list.data.apps.app,(item)=>{
        return item.state === "RUNNING";
      }).length,
      "completed": _.filter(update_yarn_job_list.data.apps.app,(item)=>{
        return item.state === "FINISHED";
      }).length,
      "failed": _.filter(update_yarn_job_list.data.apps.app,(item)=>{
        return item.state === "FAILED";
      }).length,
    }

    this.setState({
        "yarn_job_status": yarn_job_status
    });

    //console.log(yarn_job_status);
  }

  componentDidMount(){
    this.getServerState();
    this.setState({
        "server_status_intervalId": setInterval(()=>{
          this.getServerState();
        }, 3000)
    });
  }

  componentWillUnmount() {
     // use intervalId from the state to clear the interval
     if(this.state.server_status_intervalId != null){
       clearInterval(this.state.server_status_intervalId);
     }
  }

  render() {
    return (
        <div className="left_wrap pull-left">
            {/* A Menu Bar */}
            <div className="left_wrap_title">Summary</div>

            <Status
              host_starus={this.props.host_starus}
              storage_usage={this.props.storage_usage}
              yarn_job_status={this.state.yarn_job_status}
              yarn_status={this.props.yarn_status}
              />

        </div>

    );
  }
};

const mapStateToProps = (state) => {
    return {
        host_starus: state.status.host_starus,
        storage_usage: state.status.storage_usage,
        yarn_job_list: state.status.yarn_job_list,
        yarn_status: state.status.yarn_status,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
        update_host_status: (host_starus) => {
          //console.log(host_starus);
          dispatch(actions.update_host_status(host_starus))
        },
        update_storage_usage: (storage_usage) => {
          //console.log(storage_usage);
          dispatch(actions.update_storage_usage(storage_usage))
        },
        update_yarn_job_list: (yarn_job_list) => {
          //console.log(yarn_job_list);
          dispatch(actions.update_yarn_job_list(yarn_job_list))
        },
        update_yarn_status: (yarn_status) => {
          //console.log(yarn_status);
          dispatch(actions.update_yarn_status(yarn_status))
        },
    };
};


export default connect(mapStateToProps, mapDispatchToProps)(Menu);
