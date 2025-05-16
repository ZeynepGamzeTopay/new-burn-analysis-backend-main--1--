using BurnAnalysisApp.Models;
using Microsoft.EntityFrameworkCore;

namespace BurnAnalysisApp.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<Admin> Admins { get; set; }
        public DbSet<Doctor> Doctors { get; set; }
        public DbSet<BurnImage> BurnImages { get; set; }
        public DbSet<PatientInfo> Patients { get; set; }
        public DbSet<Visit> Visits { get; set; }
        public DbSet<ForumPost> ForumPosts { get; set; }
        public DbSet<Comment> Comments { get; set; }
        public DbSet<Notification> Notifications { get; set; }
        public DbSet<VoiceRecording> VoiceRecordings { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // Explicitly define primary key for PatientInfo
            modelBuilder.Entity<PatientInfo>()
                .HasKey(p => p.PatientID);

            modelBuilder.Entity<PatientInfo>()
                .HasIndex(p => p.PatientID)
                .IsUnique(); // Ensure unique PatientID

            // One-to-Many relationship between Admin and Doctor
            modelBuilder.Entity<Doctor>()
                .HasOne(d => d.Admin)
                .WithMany()
                .HasForeignKey(d => d.AdminID)
                .OnDelete(DeleteBehavior.SetNull);

            // One-to-Many relationship between Patient and BurnImage
            modelBuilder.Entity<BurnImage>()
                .HasOne(b => b.Patient)
                .WithMany()
                .HasForeignKey(b => b.PatientID)
                .OnDelete(DeleteBehavior.Cascade);

            // One-to-Many relationship between Doctor and BurnImage
            modelBuilder.Entity<BurnImage>()
                .HasOne(b => b.Doctor)
                .WithMany()
                .HasForeignKey(b => b.UploadedBy)
                .OnDelete(DeleteBehavior.SetNull);

            // Update ForumPost - Patient relationship to prevent duplicate key issues
            modelBuilder.Entity<ForumPost>()
                .HasOne(fp => fp.Patient)
                .WithMany()
                .HasForeignKey(fp => fp.PatientID)
                .OnDelete(DeleteBehavior.Restrict); // Changed from Cascade to Restrict

            // **NEW**: One-to-Many Relationship between ForumPost and Notification
            modelBuilder.Entity<Notification>()
                .HasOne(n => n.ForumPost)
                .WithMany(fp => fp.Notifications)
                .HasForeignKey(n => n.ForumPostID)
                .OnDelete(DeleteBehavior.SetNull); // Eğer bildirim silinirse forum postu silinmez
        }
    }
}
