import React, { Component } from 'react';
import { connect } from 'react-redux';
import * as actions from '../../../actions/Status';
import * as service from '../../../services/getStatus';

import AlertMenu from './AlertMenu';

class Alert extends Component{

    constructor(props){
      super(props);
      this.state = {
        "update_alert_list_intervalId": null,
      }
    }

    componentDidMount(){
      this.getAlertList();
      this.setState({
        "update_alert_list_intervalId" : setInterval(()=>{
          this.getAlertList();
        }, 5 * 1000 * 60)
      })
    }

    componentWillUnmount(){
      clearInterval(this.state.update_alert_list_intervalId);
      this.setState({
        "update_alert_list_intervalId" : null
      })
    }

    componentDidUpdate(){

    }

    getAlertList = async () => {
      const alert = await service.getAlert();
      const alert_list = alert.data.items;

      this.props.update_alert_list(
        alert_list
      );
      //console.log(alert_list);
    }


    render() {
      return(
        <AlertMenu
          alert_list={this.props.alert_list}
          />
     );
    }

}


const mapStateToProps = (state) => {
    return {
        alert_list: state.status.alert_list,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
        update_alert_list: (alert_list) => {
          //console.log(alert_list);
          dispatch(actions.update_alert_list(alert_list))
        },
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(Alert);
