

class ShowNodes extends React.Component {
	render() {
		return (
				<div>
					<Vis/>
				</div>
		)
	}
}


class Vis extends React.Component {
		
	constructor(props) {
	    super(props);
	    this.appRef = React.createRef();
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
			          shape: 'icon',
			          color: '#FF9900', // orange
			          icon: {
			              face: 'FontAwesome',
			              code: '\uf233',
			              size: 50,
			          }
			        },
			        
			        'disk' :{
				          shape: 'icon',
				          color: '#FF9900', // orange
				          icon: {
				              face: 'FontAwesome',
				              code: '\uf1c0',
				              size: 30,
				          }
				     },
			      },
//			       physics: {
//			           stabilization: true
//			       },
			      physics: false,
//			      manipulation :{
//			          enabled:true,
//			          initiallyActive:true,
//			      },
			      interaction: { multiselect: true},
			      //configure: true
			    };
			    this.network = new vis.Network(this.appRef.current, data, options);
	  }
	
	  
	// 컴포넌트가 생생될시 발생하는 이벤트 처리함수
	  componentDidMount() {
		  this.draw();
	  }
	
	  

	
	render() {
		return (
				<div className="vis"  ref={this.appRef}/>
		)
	}
}

//this.props 기본값
Vis.defaultProps = {

  name:"Compute Partition",
  node: [
          {id: 5, shape: 'icon', label: 'MDS', group:'server'  , x:0,  y:0,information:{}},
          {id: 6, shape: 'icon', label: 'OSS 1', group:'server', x:100,  y:0,information:{}},
          {id: 7, shape: 'icon', label: 'OSS 2', group:'server', x:200,  y:0,information:{}},
          {id: 8, shape: 'icon', label: 'OSS 3', group:'server', x:300,  y:0,information:{}},
          
          {id: 51, shape: 'icon', label: 'MST 1', group:'disk', x:0,  y:100,information:{}},
          
          {id: 61, shape: 'icon', label: 'OST 1', group:'disk', x:100,  y:100,information:{}},
          {id: 62, shape: 'icon', label: 'OST 2', group:'disk', x:100,  y:200,information:{}},
          {id: 63, shape: 'icon', label: 'OST 3', group:'disk', x:100,  y:300,information:{}},
          
          {id: 74, shape: 'icon', label: 'OST 4', group:'disk', x:200,  y:100,information:{}},
          {id: 75, shape: 'icon', label: 'OST 5', group:'disk', x:200,  y:200,information:{}},
          {id: 76, shape: 'icon', label: 'OST 6', group:'disk', x:200,  y:300,information:{}},
          
          {id: 87, shape: 'icon', label: 'OST 7', group:'disk', x:300,  y:100,information:{}},
          {id: 88, shape: 'icon', label: 'OST 8', group:'disk', x:300,  y:200,information:{}},
          {id: 89, shape: 'icon', label: 'OST 9', group:'disk', x:300,  y:300,information:{}},
  ],
  edges : [
	  	  {from: 5, to: 6, length: 200, width: 6, },
	  	  {from: 6, to: 7, length: 200, width: 6, },
	  	  {from: 7, to: 8, length: 200, width: 6, },
	  	  
	  	  {from: 5, to: 51, length: 200, width: 6, },
	  	  
	  	  {from: 6, to: 61, length: 200, width: 6, },
	  	  {from: 61, to: 62, length: 200, width: 6, },
	  	  {from: 62, to: 63, length: 200, width: 6, },
	  	  
	  	  {from: 7, to: 74, length: 200, width: 6, },
	  	  {from: 74, to: 75, length: 200, width: 6, },
	  	  {from: 75, to: 76, length: 200, width: 6, },
	  	  
	  	  {from: 8, to: 87, length: 200, width: 6, },
	  	  {from: 87, to: 88, length: 200, width: 6, },
	  	  {from: 88, to: 89, length: 200, width: 6, },
	  	  
  ],
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


class Main extends React.Component {
    render() {
        return (
        		<Grid>
        		  <Row className="show-grid">
        		    <Col xs={12}>
        		    	<h1>Lustre View</h1>
        		    </Col>
        		  </Row>
        		  <Row className="show-grid compute-nodes">
	      		    <ShowNodes/>
	      		  </Row>
        		 </Grid>
        );
    }
}


ReactDOM.render(
        <Main />,
        document.getElementById('root')
);