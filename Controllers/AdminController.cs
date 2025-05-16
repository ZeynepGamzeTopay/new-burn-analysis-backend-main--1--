using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using BurnAnalysisApp.Services;
using Microsoft.AspNetCore.Authorization;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;

namespace BurnAnalysisApp.Controllers
{
    [ApiController]
    [Route("api/admin")]
    public class AdminController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly IJwtService _jwtService;

        public AdminController(AppDbContext context, IJwtService jwtService)
        {
            _context = context;
            _jwtService = jwtService;
        }

        // HashPassword metodunu en üste taşıdık
        private string HashPassword(string password)
        {
            using (var sha256 = SHA256.Create())
            {
                var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
                return Convert.ToBase64String(bytes);
            }
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] Admin loginRequest)
        {
            if (string.IsNullOrWhiteSpace(loginRequest.Email) || string.IsNullOrWhiteSpace(loginRequest.Password))
                return BadRequest(new { success = false, message = "Email and Password are required." });

            string hashedPassword = HashPassword(loginRequest.Password);

            var admin = await _context.Admins
                .FirstOrDefaultAsync(a => a.Email == loginRequest.Email && a.Password == hashedPassword);

            if (admin == null)
                return Unauthorized(new { success = false, message = "Invalid credentials." });

            var token = _jwtService.GenerateToken(admin.AdminID, admin.Email);

            return Ok(new 
            { 
                success = true, 
                message = "Login successful.", 
                token,
                adminName = admin.Name, // Admin adını da frontend'e gönder
                adminEmail = admin.Email 
            });
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] Admin admin)
        {
            if (string.IsNullOrWhiteSpace(admin.Email) || string.IsNullOrWhiteSpace(admin.Password))
                return BadRequest(new { success = false, message = "Email and Password are required." });

            // Şifreyi hashleyerek kaydet
            admin.Password = HashPassword(admin.Password);

            _context.Admins.Add(admin);
            await _context.SaveChangesAsync();

            return Ok(new { success = true, message = "Admin registered successfully." });
        }

        [HttpGet("info")]
        [Authorize] // Admin'in giriş yapması gerektiğini belirtiyoruz
        public async Task<IActionResult> GetAdminInfo()
        {
            var adminIdClaim = User.FindFirst(ClaimTypes.NameIdentifier)?.Value;

            // Eğer admin ID'si mevcut değilse, Unauthorized dönüyoruz
            if (adminIdClaim == null) return Unauthorized();

            // adminId'yi int'ye çeviriyoruz çünkü veritabanında AdminID int olarak tanımlı
            if (!int.TryParse(adminIdClaim, out int adminId))
            {
                return BadRequest("Geçersiz Admin ID formatı.");
            }

            // Admin bilgilerini veritabanından alıyoruz
            var admin = await _context.Admins.FirstOrDefaultAsync(a => a.AdminID == adminId);

            // Eğer admin bulunamazsa, NotFound döndürüyoruz
            if (admin == null) return NotFound("Admin bulunamadı.");

            return Ok(new 
            { 
                name = admin.Name, 
                email = admin.Email 
            });
        }
    }
}
