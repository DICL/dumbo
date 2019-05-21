import React, { Component } from 'react';
import PlotlyJS from './PlotlyJS'
import {getYarnJobMonitor} from '../../../services/getStatus';

import * as actions from '../../../actions/Status';
import { connect } from 'react-redux';

class AppMonitor extends Component{


  constructor() {
      super();
      this.state = {
        intervalId : null,
      }
  }

  componentDidMount() {
     //var intervalId = setInterval(this.timer, 1000);
     // store intervalId in the state so it can be accessed later:
     this.setState({
       intervalId: setInterval(this.getApplicationMonitoring, 1000)
     });
  }

  componentWillUnmount() {
   // use intervalId from the state to clear the interval
   clearInterval(this.state.intervalId);
   this.setState({
     intervalId: null
   });
  }

  getApplicationMonitoring = async () => {
    // 서버에 요청하는 시간 : 3초
    const setTimeOver = 3000;
    // 기본값 세팅
    let yarn_job_monitor_list = [];
    try {
      yarn_job_monitor_list = await getYarnJobMonitor(setTimeOver);
      yarn_job_monitor_list = yarn_job_monitor_list.data;
    } catch (e) {
      console.warn('Time out getApplicationMonitoring! (3 second)');
    }

    this.props.update_yarn_job_monitor_list(
      yarn_job_monitor_list
    );
    //console.log('monitor => ',yarn_job_monitor_list.data);
  }




  render() {
    const {yarn_job_monitor_list} = this.props;
    return(
      <PlotlyJS
        data={yarn_job_monitor_list}
        />
    )
  }

}


const mapStateToProps = (state) => {
    return {
      yarn_job_monitor_list : state.status.yarn_job_monitor_list,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
      update_yarn_job_monitor_list : (yarn_job_monitor_list) => {
        dispatch(actions.update_yarn_job_monitor_list(yarn_job_monitor_list))
      }
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(AppMonitor);
