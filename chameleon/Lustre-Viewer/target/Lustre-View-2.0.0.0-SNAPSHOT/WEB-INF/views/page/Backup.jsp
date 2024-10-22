<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	<div class="col-12 mt-5">
			  		<h2>Backup</h2>
			  		<div class="col-12 border">
			  			<div class="col-12 px-3 py-3">
			  				<h5>Target OST devices</h5>
			  				<div id="backup-page">
			  					<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
				  				<div class="form-check">
									<input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
									<label class="form-check-label" for="defaultCheck1">
								    	OST1 oss1:/dev/sdb
									</label>
								</div>
			  				</div>
			  				
							
							<div class="form-group mt-5">
							    <label for="exampleInputPassword1">Backup file location</label>
							    <input type="text" class="form-control" id="file_localtion" placeholder="" value="/tmp" readonly="readonly">
							</div>
							
							<div class="col-12 px-3 py-3 text-right">
								<button type="button" id="backup-apply" class="btn btn-primary">Apply</button>
								<button type="button" id="backup-reset" class="btn btn-primary">Reset</button>
							</div>
			  			</div>
			  		</div>
			  	</div>
				
			</div>
		</div>
	</div>
</body>


<jsp:include page="commons/javascropts.jsp" flush="true"></jsp:include>
<jsp:include page="commons/modals.jsp" flush="true"></jsp:include>
</html>