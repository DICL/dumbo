import React, { Component } from 'react';
import Vis from './Vis';
import * as service from '../../services/getNodes';
import { connect } from 'react-redux';
import * as node_actions from '../../actions/Nodes';
import * as modals_actions from '../../actions/Modals';

import _ from 'lodash';

import Loading from '../Loading';
//import Modal from './Modal';


class ComputeNodes extends Component{

  _mounted = false

  ComputePartition = null
  StoragePartition = null
  Vis1 = null
  Vis2 = null
  // 선택된 노드들
  selectNodes1 = null
  selectNodes2 = null

  constructor(props) {
      super(props);
      this.state = {
        isData:false,
        "node_status_intervalId": null,
        "node_interval_list":[],
     };
  }

  componentWillMount() {
    this._mounted = true
    if (!this.state.isData && this._mounted) {

      this.setState({
          "node_status_intervalId": setInterval(()=>{
            this.getNodeList();
          }, 3000)
      });

      // this.getNodeList();
    }
    //this.getNodeList();
  }

  componentWillUnmount() {

     // use intervalId from the state to clear the interval
     if(this.state.node_status_intervalId != null && this._mounted ){
       clearInterval(this.state.node_status_intervalId);
       this.setState({
           "node_status_intervalId": null,
           "node_interval_list":[],
       });
     }
     this._mounted = false
  }

  getNodesForHosts = (node_list) =>{
    let tmp_result_data = node_list;
    //console.log(tmp_result_data);
    _.each(node_list , (item)=>{
      var host_name = item.Hosts.host_name;
      service.getNodeMatricDataArrForHost(host_name).then( request => {
        var index = _.findIndex(node_list ,( node_info )=>{ return node_info.Hosts.host_name === host_name });
        //console.log(index,request);
        let metric_data = request.data.items[0].metrics;
        tmp_result_data[index].metrics = metric_data;
        //console.log(tmp_result_data);

        this.props.update_node_list(
          tmp_result_data
        );

        this.ComputePartition = this.convert_node_list(tmp_result_data,"yarn")
        this.StoragePartition = this.convert_node_list(tmp_result_data,"lustre")
        this.StoragePartitionEdge = this.convert_lustre_node_edge(this.StoragePartition);

      }).catch( ( error ) => {
        console.log( error );
      })
    });
  };

  // ambari node 리스트들을 가져오는 메서드
  getNodeList = async () => {
      // const node_list = await service.getNodeMatricDataArr();
      const node_list = await service.getNodeMatricDataArrBak();
      let tmp_node_data = node_list.data.items;

      console.log('debug line 95:',node_list.data.items);

      this.props.update_node_list(
        node_list.data.items
      );

      //this.getNodesForHosts(node_list.data.items);





      this.ComputePartition = this.convert_node_list(node_list.data.items,"yarn")
      this.StoragePartition = this.convert_node_list(node_list.data.items,"lustre")
      this.StoragePartitionEdge = this.convert_lustre_node_edge(this.StoragePartition);

      //console.log(this.StoragePartition);
      // this.setState({
      //   isData:true,
      //   ComputePartition:this.convert_node_list(node_list.data.items,"yarn"),
      //   StoragePartition:this.convert_node_list(node_list.data.items,"lustre"),
      // });
      if(this._mounted) {
        this.setState({
          isData:true
        });
      }
      //console.log(this.convert_node_list(node_list.data.items,"lustre"));
  }





  find_mds_ndoe = (node_list) => {
    return _.find(node_list,(node_info)=>{
      let temp_find_node = _.find(node_info.information.host_components,(component_info)=>{
        // 전체 노드정보중에서 LustreMDSMgmtService 설치된 노드만 찾기
        return component_info.HostRoles.component_name === 'LustreMDSMgmtService';
      })
      return typeof temp_find_node !== 'undefined'
    });
  }

  // mds 노드를  기준으로 라인 그리기
  convert_lustre_node_edge = (node_list) => {

    let parent_node = this.find_mds_ndoe(node_list)
    let child_node = _.filter(node_list, (node_info)=>{ return node_info.label !== parent_node.label; });

    let result = _.map(child_node,(node_info)=>{
      return {
        from: parent_node.id,
        to: node_info.id,
      }
    });
    //console.log(parent_node,child_node,result);
    return result;
  }


  // 서버로부터 받은 데이터들을 visjs 형싱에 맞게 변형하는 메서드
  convert_node_list = (node_list,types)=>{
    let result = [];
    let tmplist = [];
    //console.log(node_list)
    //console.log(this.Vis1);

    // Storage Partition 노드구성
    if(types === 'lustre'){
      tmplist = _.filter(node_list,(item)=>{
        var tmpResult = false;
        _.each(item.host_components,(host_component)=>{
          if(_.isMatch(host_component.HostRoles,{"component_name":"LustreFSKernelClient"})){
            tmpResult = true;
          }
        });
        return tmpResult
      });
    // Compute Partition 노드 구성
    }else{
      tmplist = _.filter(node_list,(item)=>{
        var tmpResult = false;
        _.each(item.host_components,(host_component)=>{
          if(_.isMatch(host_component.HostRoles,{"component_name":"LustreFSKernelClient"})){
            tmpResult = true;
          }
        });
        return !tmpResult
      });


    }
    //console.log(tmplist);
    if(types === 'lustre'){
      var mds_node = _.find(tmplist,(node_info)=>{
        let is_match = _.find(node_info.host_components,(component_info)=>{ return component_info.HostRoles.component_name === 'LustreMDSMgmtService' });
        return typeof is_match !== 'undefined';
      });
      //console.log(mds_node);
    }
    result = _.map(tmplist,(item,index)=>{
        let tempMap = {
          id: index,
          label: item.Hosts.host_name,
          size: 30,
          shape: 'square', color: this.setStatusColor((item.metrics)? item.metrics.cpu.cpu_user : 0),
          //x: (index * 150), y: 0 ,
          information: item,
          metrics : item.metrics
        }

        if(types === 'lustre'){
          if(item.Hosts.host_name === mds_node.Hosts.host_name){
            tempMap.x = 0;
            tempMap.y = 0;
          }else{
            tempMap.x = 150;
            tempMap.y = ((index -1) * 150);
          }
        }else{
          tempMap.x = (index * 150);
          tempMap.y = 0;
        }

        return tempMap;
    });
    // console.log(result);
    return result;
  }

  // Vis 노드의 사룡량에 따라 색상을 변경하는 메서드
  setStatusColor = (value)=>{
    //console.log(value);
    let result = "";

    if (value > 80) {
      result = "#f5baba";
    } else if(value <= 80 && value > 60){
      result = "#ffcaa7";
    } else if(value <= 60 && value > 40){
      result = "#f4dfa1";
    } else if(value <= 40 && value > 20){
      result = "#b0e3d3";
    } else if(value <= 20 && value > 0){
      result = "#a7c4ff";
    }

    return result;
  }

  // je.kim modal 창 띄우는 방식을 노트 선택이 아닌 View Node 클릭시 실행이 되도록 수정
  viewNodes = (name) => {
    let node = null;
    if(name === "Compute Partition"){
      node = this.selectNodes
    }else{
      node = this.selectNodes2
    }


    // 선택한 노드가 없을경우 예외처리
    if(node == null) return;

    //host 만 추출
    let result = _.map(node,(item)=>{
        return item.information.Hosts.host_name
    })
    // console.log(node,result);
    this.props.create_modal(result,'default');
  }

  // Vis 노드클릭시 시작되는 메서드
  viewNodeInfomation = (node,vis,name) =>{
    //console.log(node);

    if(name === "Compute Partition"){
      this.selectNodes = node;
    }else{
      this.selectNodes2 = node;
    }



    // host 만 추출
    // let result = _.map(node,(item)=>{
    //     return item.information.Hosts.host_name
    // })
    // // console.log(node,result);
    // this.props.create_modal(result,'default');
  }


  render () {
    if (!this.state.isData) {
      // Render loading state ...
      return (
        <Loading />
      )
    } else {
      // Render real UI ...
      return (
        <div>
            <Vis
              name="Compute Partition"
              node={this.ComputePartition}
              ref={(Vis) => {this.Vis1 = Vis;}}
              viewNodeInfomation={this.viewNodeInfomation}
              setStatusColor = {this.setStatusColor}
              viewNodes = {this.viewNodes}
              />
            <Vis
              name="Storage Partition"
              node={this.StoragePartition}
              edges = {this.StoragePartitionEdge}
              ref={(Vis) => {this.Vis2 = Vis;}}
              viewNodeInfomation={this.viewNodeInfomation}
              setStatusColor = {this.setStatusColor}
              viewNodes = {this.viewNodes}
              />
        </div>
      )
    }
  }
}


const mapStateToProps = (state) => {
    return {
        node_list: state.status.node_list,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
        update_node_list: (node_list) => {
          console.log(node_list);
          dispatch(node_actions.update_node_list(node_list))
        },
        create_modal: (modal_data,component_name) => {
          //console.log(modal_data);
          dispatch(modals_actions.create_modal(modal_data,component_name))
        },
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(ComputeNodes);
