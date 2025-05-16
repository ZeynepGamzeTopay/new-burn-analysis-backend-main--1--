using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;
using BurnAnalysisApp.Data;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.EntityFrameworkCore;

namespace BurnAnalysisApp.Services
{
    public class ReminderBackgroundService : BackgroundService
    {
        private readonly ILogger<ReminderBackgroundService> _logger;
        private readonly IServiceProvider _serviceProvider;

        public ReminderBackgroundService(ILogger<ReminderBackgroundService> logger, IServiceProvider serviceProvider)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("ReminderBackgroundService is starting.");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    using (var scope = _serviceProvider.CreateScope())
                    {
                        var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                        var emailService = scope.ServiceProvider.GetRequiredService<IEmailService>();

                        // Hastaları kontrol et
                        var patients = await dbContext.Patients.ToListAsync(stoppingToken);

                        foreach (var patient in patients)
                        {
                            if (patient.HospitalArrivalDate.HasValue)
                            {
                                var reminderDate = patient.HospitalArrivalDate.Value.AddDays(7); // 1 hafta sonra
                                if (DateTime.Now >= reminderDate && !patient.ReminderSent) // E-posta gönderilmediyse
                                {
                                    await emailService.SendAppointmentReminderEmailAsync(
                                        patient.Email,
                                        patient.Name,
                                        reminderDate
                                    );

                                    // E-posta gönderildi olarak işaretle
                                    patient.ReminderSent = true;
                                    dbContext.Patients.Update(patient);
                                    await dbContext.SaveChangesAsync(stoppingToken);

                                    _logger.LogInformation($"Reminder email sent to {patient.Email}.");
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error occurred while sending reminder emails.");
                }

                // 1 saatte bir kontrol et
                await Task.Delay(TimeSpan.FromHours(1), stoppingToken);
            }

            _logger.LogInformation("ReminderBackgroundService is stopping.");
        }
    }
}