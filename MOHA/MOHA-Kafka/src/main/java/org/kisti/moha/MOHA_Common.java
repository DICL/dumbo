package org.kisti.moha;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;

public class MOHA_Common {
	public static String convertLongToDate(long dateMilisecs) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss", Locale.US);

		GregorianCalendar calendar = new GregorianCalendar(TimeZone.getTimeZone("US/Central"));
		calendar.setTimeInMillis(dateMilisecs);

		String dateFormat = sdf.format(calendar.getTime());

		return dateFormat;
	}

	public static long convertDateToLong(String dateTime) {
		long timeMillis = 0;

		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US);
		if (dateTime != null) {
			try {
				Date timeDate = sdf.parse(dateTime);
				if (timeDate != null) {
					timeMillis = timeDate.getTime();
				}

			} catch (ParseException e) {
				// TODO Auto-generated catch block
				System.out.println("Date time: " + dateTime);
				e.printStackTrace();
			}
		}
		return timeMillis;
	}
}
