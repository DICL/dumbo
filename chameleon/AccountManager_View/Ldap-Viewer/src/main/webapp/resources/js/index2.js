$(function() {
	/**
	 * jsTree에서 유저정보를 출력하는 부분
	 */
    $('#tree').jstree({
        'core' : {
            'themes': {
                'name': 'proton',
                'responsive': true
            },
            // 이함수를 통하여 Json으로 받아온다
            'data' : getData,
        } 
    }).jstree();
    getServerInfo();
    indexStatus();
});


/**
 * Ambari-view Setting 에 저정된 서버정보를 페이지에 출력하는 메서드
 */
var getServerInfo = function() {
	$.ajax({
        url: contextPath+'/ldap/serverInfo',
        success: function (data) {
        	var text = 'domain : '+data.domain+ ' dn : cn='+data.ManagerName+","+ data.dc;
        	$('#serverInfo').text(text);
        }
    });
};


/**
 * 클릭등의 이벤트를 감시하는 메서드
 */
var indexStatus = function () {
	// jsTree가 불려올경우 모든 트리메뉴을 열어놓는다.
	$('#tree').off('loaded.jstree').on('loaded.jstree', function(event, data) {
        $("#tree").jstree("open_all");
    });
	
	// jsTree 에서 노드들을 선택할경우 이벤트
    $('#tree').off('select_node.jstree').on('select_node.jstree', function (event, data) {
        loadContent(data);
    });
    // Save 이벤트
    $('.btn_save').off('click').on('click',function(){
        modifyUser();
    });
    // 사용자 추가 버튼을 누를경우
    $('.add_user').off('click').on('click',function(){
    	var uidNumber,gidNumber;
    	
    	$.ajax({
			url: contextPath+'/ldap/getMaxUidNumber',
		}).then(function(data) {
			openLayer('addUser',function(){
                uidNumber = data.result.uidNumber + 1;
                gidNumber = data.result.gidNumber;
                groupName = data.result.groupName;
                // gidNumber = data.result.gidNumber + 1;
                // gidNumber = '1004';
				$('#addUser #uidNumber').val(uidNumber);
	        	$('#addUser #gidNumber').val(gidNumber);
                $('#addUser #groupName').val(groupName);
	    	});
		});
    	
    	
    });
    // 사용자 추가 레이어팝업에서 확인버튼을 누를시
    $("#addUser .btn_add_user").off('click').on('click',function(){
		addUser($('#addUser').serializeArray());
    });
    
    // 폴더추가 버튼누를경우
    $('.create_home_folder').off('click').on('click',function () {
        createHomeFolder();
    });
    
    // 패스워드 변경
    $('.btn_change_pw').off('click').on('click',function(){
    	openLayer('changePassWord',function(){
    	});
    });
    
    // 유저삭제
    $(".btn_del").off('click').on('click',function(){
		delUser();
	});
    
    // uid 입력시 homeDirectory , sn, cn 자동입력
    $('#uid').off('change').on('change',function(){
        $('#homeDirectory').val('/home/'+$(this).val());
        $('#sn').val($(this).val());
    	$('#cn').val($(this).val());
    });
};

/**
 * 유저삭제하는 메서드
 */
var delUser = function() {
	var select_node = $('#tree').jstree('get_selected', true)[0];
	if(!select_node || $('#tree').jstree('get_selected', true).length > 1) return;
	if(!confirm('Are you sure you want to delete it?')) return;
	var dn = select_node.original.dn;
	$.ajax({
        url: contextPath+'/ldap/deleteUser',
        method: 'POST',
        headers: { 
            'Accept': 'application/json',
            'Content-Type': 'application/json' 
        },
        dataType:'json',
        data: JSON.stringify({
        	dn: dn,
        }),
        success: function (data) {
        	console.log('deleteUser',data);
            alert('User has been deleted.');
        	$('#tree').jstree(true).refresh();
        	$('#content-body .panel-body').empty();
            $('#content-body .panel-footer').empty();
            $('#content-body .panel-heading').text(' ');
        }
    });
};

/**
 * 유저추가하는 메서드
 */
var addUser = function(formData){
    if(!formData){
        return;
    }
    $.ajax({
        url: contextPath+'/ldap/addUser',
        method: 'POST',
        headers: { 
            'Accept': 'application/json',
            'Content-Type': 'application/json' 
        },
        dataType:'json',
        data: JSON.stringify(formData),
        success: function (data) {
        	console.log('addUser',data);
            alert('The user is registered.');
            $("#myModal").modal('hide');
        	$('#addUser')[0].reset();
        	$('#tree').jstree(true).refresh();
        	$('#content-body .panel-body').empty();
            $('#content-body .panel-footer').empty();
            $('#content-body .panel-heading').text(' ');
        },
        error: function(jqXHR,textStatus, errorThrown) {
        	
        	if(jqXHR.responseJSON.message){
        		alert(jqXHR.responseJSON.message);
        	}
		},
    });
};


/**
 * 유저정보를 수정하는메서드(사용안함)
 */
var modifyUser = function () {
    var formData = $('#userInfo').serializeArray();
    $.ajax({
        url: contextPath + '/ldap/modifyUser',
        headers: { 
            'Accept': 'application/json',
            'Content-Type': 'application/json' 
        },
        dataType:'json',
        method: 'POST',
        processData: false,
        contentType: false,
        data: JSON.stringify(formData),
        success: function (data) {
        	console.log('modifyUser',data);
            alert('수정되었습니다.');
            $('#tree').jstree(true).refresh();
            $('#content-body .panel-body').empty();
            $('#content-body .panel-footer').empty();
            $('#content-body .panel-heading').text(' ');
        },
    });
}

/**
 * 유저리스트를 가져오는 메서드
 */
var getData = function(obj, callback){
	var url = contextPath + "/ldap/getUserListTree";
    $.ajax({
        url: url,
        data:{},
        method:"POST",
        success:function(data){
        	callback.call(this,data.result);
        	indexStatus();
        }
    });
};

// 유저상세정보 불려오기
// data jsTree에서 선택한 정보
/**
 * 클릭등의 이벤트를 감시하는 메서드
 */
var loadContent = function(data){
    var id = data.selected[0]; //선택한 노드의 아이디
    var name = data.node.text;
    var dapth = data.node.parents.length; //선택한 노드의 깊이
    var jsonData;
    
    // 패널정보 초기화
    $('#content-body .panel-body').empty();
    $('#content-body .panel-body').spin();
    $('#content-body .panel-footer').empty();
    $('#content-body .panel-heading').text('상세보기'); //타이틀 초기화
    
    
    // 매니저 혹은 서버 그리고 유저인지 구별해서 거기에 맞는 정보출력
    switch(dapth){
        case 1:
            break;
        case 2: // 서버 혹은 매니저일떄
        	//패널 타이틀은 트리에서 선택한 이름으로
        	$('#content-body .panel-heading').text(name);
        	//서버에서 유저정보 불려오기
            $.ajax({
                url: contextPath +'/ldap/getUserInfo',
                data:{
                	id:id,
                },
            }).then(function(data){
            	jsonData = data;
            	$('#content-body .panel-body').spin();
            	//성공할시 handbars.js을 이용하여 템플릿을 불려오고 html으로 파싱
                $.get(contextPath +'/resources/template/people.hbs', function (data) {
                	Handlebars.registerHelper('ifIn', function (search, text, options) {
                        if (text.indexOf(search) > -1) {
                            return options.fn(this);
                        }
                        return options.inverse(this);
                    });
                    var template=Handlebars.compile(data);
                    $('#content-body .panel-body').html(template(jsonData));
                    indexStatus();
                }, 'html');
            });
        	break;
        case 3: //유저일경우
        	$('#content-body .panel-heading').text(name);
            $.ajax({
                url: contextPath +'/ldap/getUserInfo',
                data:{
                	id:id,
                },
            }).then(function(data){
            	jsonData = data;
            	$('#content-body .panel-body').spin();
            	//성공할시 handbars.js을 이용하여 템플릿을 불려오고 html으로 파싱
                $.get(contextPath +'/resources/template/people.hbs', function (data) {
                	Handlebars.registerHelper('ifIn', function (search, text, options) {
                        if (text.indexOf(search) > -1) {
                            return options.fn(this);
                        }
                        return options.inverse(this);
                    });
                    var template=Handlebars.compile(data);
                    $('#content-body .panel-body').html(template(jsonData));
                    // 유저일경우에는 삭제 버튼추가
                    $('#content-body .panel-footer').html(
                    		'<button class="btn btn-danger pull-right btn_del" type="button">Delete</button>'
//                    		+'<button class="btn btn-warning pull-right btn_change_pw" type="button" style="margin-right: 7px;">Change Password</button>'
//                    		+'<button class="btn btn-warning pull-right btn_save" type="button" style="margin-right: 7px;">Save</button>'
                    );
                    indexStatus();
                }, 'html');
            });
            break;
    }
};

/**
 * 레이어 팝업생성
 * templateFile HandBars Template 저장파일명
 */
var openLayer = function(templateFile,callback) {
	if(!templateFile || !callback) return;
	
	$('#myModal').empty();
	
	var option = {
		backdrop: 'static',
		keyboard: false
	}
	
	$.get(contextPath +'/resources/template/'+templateFile+'.hbs', function (data) {
        var template=Handlebars.compile(data);
        $('#myModal').html(template());
        $('#myModal').modal(option);
        if(callback){
        	callback();
        }
        indexStatus();
    }, 'html');
};

/**
 * 홈디렉토리 생성
 */
var createHomeFolder = function () {
    var seleted_node = $('#tree').jstree('get_selected')[0].split('=');
    if(seleted_node[0] == "uid"){
        id = seleted_node[1];
    }else{
        return;
    }
    $.ajax({
        url: contextPath +'/ldap/craetedHadoopHomeFolder',
        data:{	'id':id  },
    }).then(function(data){
        console.log(data);
        alert("The Work folder has been created.");
    });
};