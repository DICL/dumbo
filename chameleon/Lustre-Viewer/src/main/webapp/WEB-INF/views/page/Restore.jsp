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
					<h2>Restore</h2>
					<div class="col-12 border">
						<div class="col-12 px-3 py-3">
							<h5>Backup file location</h5>
							<div class="form-group">
							    <input type="text" class="form-control" id="backup-file-location" placeholder="" value="/tmp" readonly="readonly">
							</div>
						</div>
						<div class="col-12 px-3 py-3">
							<h5>Backup status and device location to restore</h5>
							<div id="Restore-page">
								<div class="form-check mb-3">
								  <input class="form-check-input" type="checkbox" value="" id="defaultCheck1">
								  <label class="form-check-label" for="defaultCheck1">
								    OST1 oss1:/dev/sdb oss1-ost1-181010.tar.gz
								  </label>
								</div>
								<div class="form-group">
								    <select id="" class="form-control col-3 ml-3">
								    	<option>oss1:/dev/sdf</option>
								    </select>
								</div>
								<div class="form-group">
								    <select id="" class="form-control col-3 ml-3">
								    	<option>oss1:/dev/sdf</option>
								    </select>
								</div>
							
							</div>
						</div>
						<div class="col-12 px-3 py-3">
							<h5>Backup location</h5>
							<div class="form-group">
								<input type="text" class="form-control" id="backup-location" placeholder=""  value="/tmp" readonly="readonly">
							</div>
						</div>
						
						<div class="col-12 px-3 py-3 text-right">
							<button type="button" id="restore-apply" class="btn btn-primary">Restore</button>
							<button type="button" id="restore-reset" class="btn btn-primary">Reset</button>
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