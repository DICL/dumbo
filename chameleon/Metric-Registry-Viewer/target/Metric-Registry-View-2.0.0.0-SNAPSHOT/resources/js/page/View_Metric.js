$(function() {
	View_Metric.init();
})

var View_Metric = {
	init : function() {
		View_Metric.listener();
	},
	updateMetric : function(){
		var data = {};
		data.num = $('#num').val();
		data.name = $('#name').val();
		data.description = $('#description').val();
		data.pid_symbol = $('#pid_symbol').val();
		data.y_axis_label = $('#y_axis_label').val();
		data.parser_script = $('#parser_script').val();
		
		return $.ajax({
			url: contextPath+'/api/v1/metric/updateMetric',
			type: "POST",
			data: data
		});
	},
	listener : function() {
		$('#update_metric').off('click').on('click',function(){
			showLoading('Update Metric .....');
			View_Metric.updateMetric().then(
				function(data, textStatus, jqXHR) {
					hideLoading();
					alert("Metric information has been modified.");
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		});
	}
}