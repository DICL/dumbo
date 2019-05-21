import React, { Component } from 'react';
import Draggable from "../jqueryui-draggable";

import Default from "./default";
import Vconsole from "./vconsole";
import ApplicationHistory from './ApplicationHistory';
import YarnJobHistoryPerNode from './YarnJobHistoryPerNode';

class Modal extends Component{

  // this.props 기본값
  static defaultProps = {
    onRemove : function (idx) {
      console.warn('undefined onRemove function',idx);
    }
  }


  components = {
      'default': Default,
      'vconsole':Vconsole,
      'ApplicationHistory': ApplicationHistory,
      'YarnJobHistoryPerNode': YarnJobHistoryPerNode,
  };

  render () {
    const { modal_list, onRemove, create_modal } = this.props;
    // console.log(modal_list);

    const modalList = modal_list.map(
      // 클릭시 redux 에 있는 modal_list 에 겍체를 삽입하고
      // redux 에 있는 node_list 에서 일치하는 호스트만 추출하여 this.props 에 적재
      // 또한 동적으로 컨포넌트를 생성
      ({idx, data , component_name}) => {
        const size = data.size;
        const TagName = this.components[component_name || 'default'];
        return(
          <Draggable
            key={idx}
            size={size}
            >
              <TagName
                key={idx}
                data={data}
                idx={idx}
                onRemove={onRemove}
                create_modal={create_modal}
              />
          </Draggable>

        )
      }
    );


    return (
        <div>
          {modalList}
        </div>
    )
  }
}

export default Modal
