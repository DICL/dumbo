import React, { Component } from 'react';
import { Select,Button } from 'semantic-ui-react'
import * as service from '../../../services/getNodes';
import _ from 'lodash';
import Datetime from 'react-datetime';
import 'react-datetime/css/react-datetime.css';
import 'moment/locale/ko';
import './YarnTimeHistoryView.css'


class YarnTimeHistoryView extends Component{

    // ambari compoent 중에서 YARNAppMonitorClient 가 설치된 노드만 추출하는 메서드
    getYARNAppMonitorClientNodes = async () => {
      const result_getYARNAppMonitorClientNodes = await service.getYARNAppMonitorClientNodes();
      let node_list = [];
      if(result_getYARNAppMonitorClientNodes.status === 200 && result_getYARNAppMonitorClientNodes.data.items.length > 0){
        node_list = _.map(result_getYARNAppMonitorClientNodes.data.items,(host_info)=>{
          let temp_hostname = host_info.Hosts.host_name;
          return {
            key: temp_hostname, value: temp_hostname, text: temp_hostname
          }
        });
      }else{
        node_list = ['not found'];
      }

      // let result = {
      //   "isFetching": false,
      //   "multiple": false,
      //   "search": false,
      //   "searchQuery": null,
      //   "value": node_list[0].value,
      //   "options": node_list
      // }
      let result = node_list;
      this.setState({
          isData:true,
          YARNAppMonitorClientNodes:result,
          node: result[0].value,
      });
      console.log(node_list,result);
    }

    // 노드 변경시 처리되는 이벤트 메서드
    nodeChangeEventHandler = (event,data)=>{
      //console.log(event,data);
      this.setState({node: data.value});
    }

    // start time 텍스트창 변경시 처리
    startTimeChangeEventHandler = (event)=>{
      console.log(event);
      try {
        let result = new Date(event);
        this.setState({start_time: result});
      } catch (e) {
        console.error(e);
      }

    }
    // start time 달력 선택시 처리
    startTimeChangeEventHandler2 = (event)=>{
      console.log(event);
      if(event._d){
        this.setState({start_time: event._d});
      }

    }
    // end time 텍스트창 변경시 처리
    endTimeChangeEventHandler = (event)=>{
      console.log(event);
      try {
        let result = new Date(event);
        this.setState({end_time: result});
      } catch (e) {
        console.error(e);
      }
    }
    // end time 달력 선택시 처리
    endTimeChangeEventHandler2 = (event)=>{
      console.log(event);
      if(event._d){
        this.setState({end_time: event._d});
      }
    }

    // 모달창 생성 달력 선택시 처리
    viewTimeHistory = (event) => {
      let send_data = {
        node : this.state.node,
        start_time : this.state.start_time,
        end_time : this.state.end_time,
      }
      this.props.viewTimeHistory(send_data);
    }

    constructor(props){
      super(props);
      let end_time = new Date();
      let start_time = new Date(new Date().setHours(end_time.getHours() - 1));
      this.state = {
        isData:false,
        YARNAppMonitorClientNodes:[],
        node:null,
        start_time: start_time,
        end_time: end_time,
      };
    }

    componentDidMount(){
      this.getYARNAppMonitorClientNodes();
    }

    componentWillUnmount(){

    }

    componentDidUpdate(){

    }



    render() {

      if (this.state.isData === false) {
        return(<div>
        </div>);
      }else{
        return(
          <div id="YarnTimeHistoryView" className="ui two column grid">
            <div className="column">
              <div className="ui two column grid">
                <div className="column">
                  <label htmlFor="node">
                    Node :
                  </label>
                  <Select
                    id="node"
                    placeholder='Select Node'
                    options={this.state.YARNAppMonitorClientNodes}
                    value={this.state.node}
                    onChange={this.nodeChangeEventHandler}

                    />
                </div>
                <div className="column">
                  <label htmlFor="node">
                    Start Time :
                  </label>
                  <div className='datetime ui input'>
                    <Datetime
                      dateFormat="YYYY-MM-DD"
                      timeFormat="HH:mm:ss"
                      locale="ko-kr"
                      value={this.state.start_time}
                      onBlur={this.startTimeChangeEventHandler}
                      onChange={this.startTimeChangeEventHandler2}
                      />
                  </div>
                </div>
              </div>

            </div>
            <div className="column">
              <div className="ui two column grid">
                <div className="column">
                  <label htmlFor="node">
                    End Time :
                  </label>
                  <div className='datetime ui input'>
                    <Datetime
                      dateFormat="YYYY-MM-DD"
                      timeFormat="HH:mm:ss"
                      locale="ko-kr"
                      value={this.state.end_time}
                      onBlur={this.endTimeChangeEventHandler}
                      onChange={this.endTimeChangeEventHandler2}
                      />
                  </div>
                </div>
                <div className="column">
                  <Button
                    id='get_history'
                    content='Get History'
                    onClick={this.viewTimeHistory}
                    />
                </div>
              </div>
            </div>
          </div>
       );
      }

    }

  }

YarnTimeHistoryView.defaultProps = {
    viewTimeHistory: ()=>{
      console.warn('Not viewTimeHistory !!!');
    }
};

export default YarnTimeHistoryView;
