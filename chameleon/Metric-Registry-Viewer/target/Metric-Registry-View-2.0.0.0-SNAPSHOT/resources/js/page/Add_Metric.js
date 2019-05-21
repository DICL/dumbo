$(function(){
	Add_Metric.init();
});

var Add_Metric = {
		init : function() {
			Add_Metric.listener();
		},
		register_metric : function (){
			var data = {};
			data.name = $('#name').val();
			data.description = $('#description').val();
			data.pid_symbol = $('#pid_symbol').val();
			data.y_axis_label = $('#y_axis_label').val();
			data.parser_script = $('#parser_script').val();
			
			return $.ajax({
				url: contextPath+'/api/v1/metric/addMetric',
				type: "POST",
				data: data
			});
		},
		listener : function() {
			$('#register_metric').off('click').on('click',function(){
				showLoading('Create Metric .....');
				Add_Metric.register_metric().then(
					function(data, textStatus, jqXHR) {
						hideLoading()
						alert("Metric information registered.");
						window.location = contextPath;
					},
					function(jqXHR,textStatus,errorThrown) {
						console.error(jqXHR,textStatus,errorThrown);
					}
				)
			});
		}
}