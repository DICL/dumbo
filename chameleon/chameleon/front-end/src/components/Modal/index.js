import React, { Component } from 'react';
import ReactModalList from './ReactModal/Components';
import { connect } from 'react-redux';
import * as modals_actions from '../../actions/Modals';

class Modal extends Component{
  constructor(props) {
    super(props);
    this.state = {
      modal_list : [
        // { id: 0, text: ' tset1' },
        // { id: 1, text: ' tset2' },
        // { id: 2, text: ' tset3' }
      ]
    }
  }

  componentWillUnmount() {
    // 컴포넌트가 사라질 때 인스턴스 제거
    this.props.truncate_modal();
  }


  render () {
    const { modal_list,remove_modal,create_modal,mouse_position  } = this.props;
    //console.log(modal_list);
    //console.log(this.props.node_list);
    return (
      <div>
        <ReactModalList
          modal_list={modal_list}
          onRemove={remove_modal}
          create_modal={create_modal}
          mouse_position={mouse_position}
          />
      </div>

    )
  }
}


const mapStateToProps = (state) => {
    return {
        modal_list: state.modals.modal_list,
    };
};

const mapDispatchToProps = (dispatch) => {
    //return bindActionCreators(actions, dispatch);
    return {
      remove_modal: (index) => {
        //console.log(node_list);
        dispatch(modals_actions.remove_modal(index))
      },

      truncate_modal: () => {
        //console.log('truncate_modal');
        dispatch(modals_actions.truncate_modal())
      },

      create_modal: (modal_data,component_name) => {
        //console.log(modal_data,component_name);
        dispatch(modals_actions.create_modal(modal_data,component_name))
      },
    };
};



export default connect(mapStateToProps, mapDispatchToProps) (Modal)
