public interface IEmailService
{
    Task SendVerificationEmailAsync(string doctorEmail, string doctorName);
    Task SendAppointmentReminderEmailAsync(string patientEmail, string patientName, DateTime appointmentDate);
}