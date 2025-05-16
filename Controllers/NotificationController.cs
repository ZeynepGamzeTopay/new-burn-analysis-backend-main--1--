using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;

namespace BurnAnalysisApp.Controllers
{
    [Route("api/notifications")]
    [ApiController]
    [Authorize] // Kullanıcı giriş yapmış olmalı
    public class NotificationController : ControllerBase
    {
        private readonly AppDbContext _context;

        public NotificationController(AppDbContext context)
        {
            _context = context;
        }

        // Doktorun bildirimlerini çek
        [HttpGet]
        public async Task<IActionResult> GetNotifications()
        {
            var doctorId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value);

            var notifications = await _context.Notifications
                .Where(n => n.DoctorID == doctorId)
                .OrderByDescending(n => n.CreatedAt)
                .ToListAsync();
            Console.WriteLine($"👀backend Bildirim Sayısı: {notifications.Count}");


            return Ok(notifications);
        }


        // Bildirimi okundu olarak işaretle
        [HttpPut("{id}/read")]
        public async Task<IActionResult> MarkAsRead(int id)
        {
            var notification = await _context.Notifications.FindAsync(id);
            if (notification == null)
                return NotFound();

            notification.IsRead = true;
            await _context.SaveChangesAsync();

            return NoContent();
        }
    }
}
