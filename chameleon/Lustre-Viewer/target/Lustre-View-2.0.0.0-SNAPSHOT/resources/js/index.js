$(function(){
	IndexPage.init();
});

var IndexPage = {
		target : "root",
		loading : null,
		init : function() {
			IndexPage.checked_tables(IndexPage.grid_lustre_nodes);
		},
		
		grid_lustre_nodes : function(callback) {
			axios.get(contextPath+'/api/v1/ambari/getLustreNodes')
			.then(function (response) {
				var data = response.data;
				if(data.length == 0){
					IndexPage.syncLustreTable(IndexPage.grid_lustre_nodes);
				}else{
					
				}
				IndexPage.stop_loading()
				IndexPage.templete_html($("#lustre-info")[0],'/template/setting_page.hbs', data, function() {
					  
				});
				
			 })
			 .catch(function (error) {
			    console.log(error);
			 });
		},
		
		syncLustreTable : function(callback) {
			IndexPage.change_loding_text("Get Ambari Server");
			axios.get(contextPath+'/api/v1/ambari/syncLustreTable')
			.then(function (response) {
				var data = response.data;
				try {
					  callback();
					} catch (e) {
						console.warn(e);
					}
			 })
			 .catch(function (error) {
			    console.log(error);
			 });
		},
		
		
		checked_tables : function(callback) {
			
			IndexPage.templete_html($("#root")[0], '/template/index.hbs', {}, function() {
				var target = "lustre-info";
				  IndexPage.loading_page(target,"check table");
				  IndexPage.check_created_tables(callback);
			});
		},
		
		loading_page : function(target_name,message) {
			
			var html = "<div class=\"loding-container\"  >" 
				+"<div id=\"loading\"></div>"
				+"<p class=\"loading-text\" >"+message+"</p>"
				+"</div>";
			
			$("#"+target_name).append(html)
			var target = document.getElementById("loading");
			IndexPage.loading = Spinner({color:'#000', lines: 12}).spin(target);
		},
		
		stop_loading : function() {
			IndexPage.loading.stop();
			$(".loding-container").remove();
		},
		
		change_loding_text : function(text) {
			$('.loading-text').text(text);
		},
		
		templete_html : function(target_class,url,data,callback) {
			axios.get(contextPath + url)
			  .then(function (response) {
				  var template=Handlebars.compile(response.data);
				  target_class.innerHTML=template(data);
				  try {
					  callback();
					} catch (e) {
						console.warn(e);
					}
			  })
			  .catch(function (error) {
			    console.log(error);
			  });
		},
		
		check_created_tables : function(callback) {
			axios.get(contextPath+'/api/v1/lustre/chechCreatedTables')
			.then(function (response) {
//					try {
//						IndexPage.stop_loading();
//					} catch (e) {
//						
//					}
				  if(!response.data.status){
					  alert("테이블 생성 오류");
				  }else{
					  try {
						  callback();
						} catch (e) {
							console.warn(e);
						}
					  
				  }
			 })
			 .catch(function (error) {
			    console.log(error);
			 });
		},
		
};