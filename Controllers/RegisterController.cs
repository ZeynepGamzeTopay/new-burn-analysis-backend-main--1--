using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.IdentityModel.Tokens; // Bu ve diğer 'using' direktifleri doğru sıraya yerleştirilmeli
using System.IdentityModel.Tokens.Jwt;
using System;

namespace BurnAnalysisApp.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RegisterController : ControllerBase
    {
        private readonly AppDbContext _context;
        private readonly string _jwtSecretKey = "s8THyJk0fgOT7p;mRw6bnvn-vUHM5Oaw"; // JWT Secret key (environment variables is a better option for production)

        public RegisterController(AppDbContext context)
        {
            _context = context;
        }

        [HttpPost]
        public async Task<IActionResult> Register([FromBody] DoctorRegisterDto model)
        {
            if (model == null || string.IsNullOrWhiteSpace(model.Email) || string.IsNullOrWhiteSpace(model.Password))
            {
                return BadRequest(new { message = "Email ve Şifre zorunludur." });
            }

            // Email kontrolü
            var existingDoctor = await _context.Doctors.FirstOrDefaultAsync(d => d.Email == model.Email);
            if (existingDoctor != null)
            {
                return BadRequest(new { message = "E-posta zaten kayıtlı." });
            }

            // Admin kontrolü
            var admin = await _context.Admins.FirstOrDefaultAsync();
            if (admin == null)
            {
                return BadRequest(new { message = "Admin bulunamadı." });
            }

            // Şifreyi hashleme
            var hashedPassword = HashPassword(model.Password);

            // Yeni doktor oluşturuluyor
            var newDoctor = new Doctor
            {
                Name = model.Name,
                Email = model.Email,
                Password = hashedPassword,
                Verified = false, // Admin onayı bekliyor
                AdminID = admin.AdminID 
            };

            _context.Doctors.Add(newDoctor);
            await _context.SaveChangesAsync();

            // Token oluşturuluyor
            var token = GenerateToken(newDoctor);

            // Kayıt başarılı ve token döndürülüyor
            return Ok(new { message = "Kayıt başarılı! Admin onayı bekleniyor.", token });
        }

        // Şifreyi SHA256 ile hash'le
        private string HashPassword(string password)
        {
            using (var sha256 = SHA256.Create())
            {
                var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
                return Convert.ToBase64String(bytes);
            }
        }

        // JWT Token üretme
        private string GenerateToken(Doctor doctor)
        {
            var securityKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_jwtSecretKey));
            var credentials = new SigningCredentials(securityKey, SecurityAlgorithms.HmacSha256);

            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new System.Security.Claims.ClaimsIdentity(new[]
                {
                    new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Name, doctor.Name),
                    new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Email, doctor.Email),
                    new System.Security.Claims.Claim("DoctorId", doctor.DoctorID.ToString())
                }),
                Expires = DateTime.UtcNow.AddDays(1), // Token'ın geçerlilik süresi
                SigningCredentials = credentials
            };

            var tokenHandler = new JwtSecurityTokenHandler();
            var token = tokenHandler.CreateToken(tokenDescriptor);
            return tokenHandler.WriteToken(token);
        }
    }

    // Register için DTO (Data Transfer Object)
    public class DoctorRegisterDto
    {
        public string? Name { get; set; }
        public string Email { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;
    }
}
