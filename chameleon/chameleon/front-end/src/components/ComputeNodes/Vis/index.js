import React, { Component, createRef } from 'react';
import vis from 'vis';
import 'vis/dist/vis.css';

import { Form, Radio, Button , Checkbox } from 'semantic-ui-react'
import './ComputeNodes.css';

import _ from 'lodash';

    // initialize your network!




// vis.js 을 통한 컴퓨터 노드를 표현하는 컴포넌트
class ComputeNodes extends Component {
  _mounted = false

  // vis.js 선언시 저장할 겍체
  network = null;
  // vis.js 에서 노드정보 저장용 겍체
  nodes = []; vis_nodes = null;
  // vis.js 에서 노드간에 연결선을 저장할 겍체
  edges = []; vis_edges = null;

  constructor(props) {
    super(props);
    // vis.js 선언시 저장할 겍체초기화
    this.network = {};
    // 해당 virtual dom 에 대한 겍체지정
    this.appRef = createRef();

    // React State
    this.state = {
     //states
     alarmCriteria : "cpu", // 현재 라디오버튼에서 체크한 이름
     modify_nodes : false, // Modify Nodes 체크박스 체크여부
    };

  }

  // 190109 je.kim 체크박스 체크시 노드를 옮길수 있는 로직 추가
  set_modify_node = (e) => {
    // state 값을 변경
    let is_modify = !this.state.modify_nodes;
    // visjs 미선언시 예외처리
    if(this.network == null){
      return;
    }
    // visjs 기본 환경설정 정보 선언
    let options = {
      interaction: {
        multiselect: true, // 다중선택 가능여부
        // 190109 je.kim zoom , drag disabled
        dragView : false, // visjs 배경 이동가능 여부
        zoomView : false, // visjs 마우스 휠로 줌인 중아웃 가능 여부
        hover: true, // visjs 노드에 마우스 오버시 색상 변경여부
        dragNodes: false, // visjs 노드를 이동가능 여부
        //navigationButtons : true,
      },
    };

    // 체크박스 선택시
    if(is_modify === true){
      // visjs 노드 이동가능하게 설정
      options.interaction.dragNodes= true;
    // 체크박스 헤제시
    }else{

      // visjs 노드들을 이동 못하게 설정
      options.interaction.dragNodes= false;
    }

    // visjs 옵션 재설정
    this.network.setOptions(options);

    // state 값을 변경하여 저장
    this.setState({
      modify_nodes : is_modify,
    })
  }



  // this.props 기본값
  static defaultProps = {
    name:"Compute Partition",
    node: [
            {id: 1, shape: 'icon', label: 'Node 1', group:'server'  ,x: -200, y: 0 ,information:{}},
            {id: 2, shape: 'icon', label: 'Node 2', group:'server'  ,x: -100, y: 0 ,information:{}},
            {id: 3, shape: 'icon', label: 'Node 3', group:'server'  ,x: 0, y: 0 ,information:{}},
            {id: 4, shape: 'icon', label: 'Node 4', group:'server'  ,x: 100, y: 0 ,information:{}},
            {id: 5, shape: 'icon', label: 'Node 5', group:'server'  ,x: 200, y: 0 ,information:{}}
    ],
    edges : [],
    // 부모에서 자식에게 전달할 예정인 클릭시 이벤트 함수
    viewNodeInfomation : (clickedNodes)=>{
      console.log('viewNodeInfomation Function is undefined clicked nodes:' , clickedNodes);
    },
    // 부모에서 전송될 예정인 컴포넌트 업데이트시 실행될 이벤트함수
    setStatusColor : (value)=>{
      console.log('setStatusColor Function is undefined ');
      return "#000000";
    }
  }

  // 초기에 Vis 노드를 그리는 메서드
  draw(){
    this.nodes = this.props.node;
    this.edges = this.props.edges;

    this.vis_nodes = new vis.DataSet(this.nodes);
    // create an array with edges
    this.vis_edges = new vis.DataSet(this.edges);


    var data = {
        nodes: this.vis_nodes,
        edges: this.vis_edges
    };
    var options = {
      groups:{
        'server' :{
          shape: 'square',
          color: '#FF9900' // orange
        }
      },
      // physics: {
      //     stabilization: true
      // },
      physics: {
        enabled : false
      },
      manipulation :{
          enabled:false,
          initiallyActive:false,
      },
      interaction: {
        multiselect: true,
        // 190109 je.kim zoom , drag disabled
        dragView : false,
        zoomView : false,
        hover: true,
        dragNodes: false,
        //navigationButtons : true,
      },
      //configure: true
    };
    this.network = new vis.Network(this.appRef.current, data, options);
  }

  // Alarm Criteria 내의 라디오 버튼을 변경시 생기는 이벤트
  handleChange = (e, { value }) => {
    if(this._mounted){
      this.setState({ alarmCriteria : value })
    }
  }

  // 각노드의 색상을 변경
  setNodeColor = (nodes)=>{
    // console.log(this.nodes);
    // state.alarmCriteria 라디오 버튼에서 체크한 항목이 저장되어 있음
    var alarmCriteria = this.state.alarmCriteria
    //180914 je.kim 노드 위치 제거
    _.each(nodes.map((node)=>{ delete node.x;  delete node.y; return node;}),(item,index)=>{
      var value = 0;
      var result = "";
      switch (alarmCriteria) {
        case "cpu":
            // cpu 정보만 있을경우에만 반영
            var cpu = (item.metrics) ? item.metrics.cpu.cpu_user : 0;
            value = cpu;
          break;
        case "mem":
            // mem 정보만 있을경우에만 백분률로 계산
            var mem = (item.metrics) ? (((item.metrics.memory.mem_total - item.metrics.memory.mem_free) / item.metrics.memory.mem_total) * 100) : 0
            value = mem;
          break;
        case "disk":
            // disk 정보만 있을경우에만 백분률로 계산
            var disk  = (item.metrics) ? (((item.metrics.disk.disk_total - item.metrics.disk.disk_free) / item.metrics.disk.disk_total) * 100) : 0
            value = disk;
          break;
        default:
          // 없을경우 cpu 정보로 수정
          value = (item.metrics) ? item.metrics.cpu.cpu_user : 0;
      }

      result = this.props.setStatusColor(value);
      item.color = result;
      this.vis_nodes.update(item);
    })

  }

  // 노드들을 가로 정렬해주는 메서드
  set_Horizontal_Alignment = (e)=>{

    // 현재 출력된 노드들의 좌표값을 저장
    let node_position = _.map(this.network.body.nodes,(node_info)=>{
      let temp = {
        x : node_info.x,
        y : node_info.y,
      };
      return temp;
    });


    // 위치수정할 임시변수 저장
    let modify_nodes = _.map(this.vis_nodes._data,(node_info,index)=>{
      node_info.x = node_position[index].x; // x축은 변동사항 없음
      node_info.y = 0; // y 축은 0으로 고정
      return node_info;
    })

    //console.log(e,this.vis_nodes,this.network,modify_nodes);
    // 수정된 정보들을 vis.js 에 반영
    this.vis_nodes.update(modify_nodes);
  }

  // 노드들을 세로 정렬해주는 메서드
  set_Vertical_Alignment = (e)=>{

    // 현재 출력된 노드들의 좌표값을 저장
    let node_position = _.map(this.network.body.nodes,(node_info)=>{
      let temp = {
        x : node_info.x,
        y : node_info.y,
      };
      return temp;
    });


    // 위치수정할 임시변수 저장
    let modify_nodes = _.map(this.vis_nodes._data,(node_info,index)=>{
      node_info.x = 0; // x축은 0으로 고정
      node_info.y = node_position[index].y; // y축은 변동사항 없음
      return node_info;
    })

    //console.log(e,this.vis_nodes,this.network,modify_nodes);
    // 수정된 정보들을 vis.js 에 반영
    this.vis_nodes.update(modify_nodes);
  }



  // 컴포넌트가 생생될시 발생하는 이벤트 처리함수
  componentDidMount() {
    this._mounted = true
    this.draw();

    // 클릭시 이벤트 처리
    var node = this.props.node
    var name = this.props.name
    var viewNodeInfomation = this.props.viewNodeInfomation;
    var network = this.network

    // 노드 선택시 이벤트 처리
    this.network.on("selectNode", function (params) {
      var ids = params.nodes;
      var clickedNodes = _.filter(node,(item,index)=>{
        return _.indexOf(ids,item.id) !== -1;
      });
      //var clickedNodes = this.state.nodes.get(ids);
      viewNodeInfomation(clickedNodes,network,name);
    })

    // 노드 비활성화시 이벤트
    this.network.on("deselectNode", function (params) {
      //console.log('test',params);
      viewNodeInfomation(null,network,name);
    });

  }

  // 컴포넌트에 데이터가 수정될경우 이벤트 처리함수 및 노드 색상변경
  componentDidUpdate(){
    //console.log(this.props.node);
    this.setNodeColor(this.props.node)
  }

  // 컴포넌트 제거시 이벤트 처리
  componentWillUnmount() {
    // 컴포넌트가 사라질 때 인스턴스 제거
    if (this.network !== null) {
      this.network.destroy();
      this.network = null;
    }
    this._mounted = false
  }




  render() {
    // console.log(this.props.node);
    return (
      <div className="right_wrap pull-left" >
        <div className="title_wrap">
          <div className="text">{this.props.name}</div>
        </div>
        <div className="contents_wrap">
          <div className="cluster_info">
            <div className="ComputeNodes">
              <div>
                <Form>
                  <Form.Field>
                    Alarm Criteria:
                  </Form.Field>
                  <Form.Field>
                    <Radio
                      label='CPU'
                      name='criteria'
                      value='cpu'
                      checked={this.state.alarmCriteria === 'cpu' }
                      onChange={this.handleChange}
                    />
                  </Form.Field>
                  <Form.Field>
                    <Radio
                      label='Mem'
                      name='criteria'
                      value='mem'
                      checked={this.state.alarmCriteria === 'mem'}
                      onChange={this.handleChange}
                    />
                  </Form.Field>

                  <Form.Field>
                    <Radio
                      label='Disk'
                      name='criteria'
                      value='disk'
                      checked={this.state.alarmCriteria === 'disk'}
                      onChange={this.handleChange}
                    />
                  </Form.Field>
                  {
                    /*
                    <Form.Field>
                      <Radio
                        label='Network'
                        name='criteria'
                        value='network'
                        checked={this.state.alarmCriteria === 'network'}
                        onChange={this.handleChange}
                      />
                    </Form.Field>
                    */
                  }

                </Form>
              </div>
              <div>
                <Form>
                  <Form.Field>
                    Alignment:
                  </Form.Field>
                  {/* 가로정렬버튼 */}
                  <Button
                    primary
                    onClick={this.set_Horizontal_Alignment}
                    >Horizontal</Button>
                  {/* 세로정렬버튼 */}
                  <Button
                    secondary
                    onClick={this.set_Vertical_Alignment}
                    >Vertical</Button>

                  {/* 노드위치 수정영 체크박스 */}
                  <Checkbox
                    label='Modify Nodes'
                    className="modify-node"
                    checked={this.state.modify_nodes}
                    onChange={this.set_modify_node}
                    />
                  {/* 해당노드의 상세정보창을 출력하는 버튼 */}
                  <Button
                    primary
                    className="view-node"
                    onClick={(e)=>{this.props.viewNodes(this.props.name);}}>
                    View Node
                  </Button>
                </Form>
              </div>
              <div className="server-information" >
                <div className="legend-used" >0 ~ 19% :</div>
  							<div className="gpu-unit-sample status-0"></div>
  							<div className="legend-used" >20 ~ 39% :</div>
  							<div className="gpu-unit-sample status-1"></div>
  							<div className="legend-used" >40 ~ 59% :</div>
  							<div className="gpu-unit-sample status-2"></div>
  							<div className="legend-used" >60 ~ 79% :</div>
  							<div className="gpu-unit-sample status-3"></div>
  							<div className="legend-used" >80 ~ 99% :</div>
  							<div className="gpu-unit-sample status-4"></div>
  							<div className="legend-used" >100% :</div>
                <div className="gpu-unit-sample status-5"></div>
              </div>
              {/* ref 로 DOM 겍체 직접 접근 */}
              <div className="vis" ref={this.appRef}/>
            </div>
          </div>
        </div>
      </div>


    );
  }
}

export default ComputeNodes;
