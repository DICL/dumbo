import React, { Component, createRef } from 'react';

// jquery & jqueryui
import $ from 'jquery';
import 'jquery-ui-bundle';
import 'jquery-ui-bundle/jquery-ui.min.css';
import './jquery-ui-bundle.css';
// jquery-ui 을 이용한 modal 창 이동처리

class Draggable extends Component {
  constructor(props){
    super(props);
    this.appRef = createRef();
  }

  componentDidMount(){
    // const {minHeight , minWidth} = this.props.size;
    this.$node = $(this.appRef.current);
    this.$node
    .draggable({
      handle: ".card .header"
    })
    // .resizable({
    //   minHeight: minHeight,
    //   minWidth: minWidth
    // })
    ;
  }

  componentWillUnmount(){
    this.$node
    .draggable( "destroy" )
    // .resizable( "destroy")
    ;
  }

  render () {
    // const {width, height} = this.props.size;
    // style={{
    //     width : width,
    //     height : height,
    //   }}

    return (
      <div
        ref={this.appRef}
        className="resizable"
        style={
          // je.kim 높이를 스크롤 기준으로 조정
          {
            top : window.scrollY - 100,
            left : 0
          }
        }
        >
        {this.props.children}
      </div>
    )
  }
}

Draggable.defaultProps = {
  size : {
    minHeight : 500,
    minWidth : 200,
    width : 200,
    height : 150,
  },
}

export default Draggable
